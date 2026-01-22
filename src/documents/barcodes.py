from __future__ import annotations

import datetime
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings
from pdf2image import convert_from_path
from pikepdf import Page
from pikepdf import PasswordError
from pikepdf import Pdf

from documents.converters import convert_from_tiff_to_pdf
from documents.data_models import ConsumableDocument
from documents.data_models import DocumentMetadataOverrides
from documents.models import Document
from documents.models import Tag
from documents.plugins.base import ConsumeTaskPlugin
from documents.plugins.base import StopConsumeTaskError
from documents.plugins.helpers import ProgressManager
from documents.plugins.helpers import ProgressStatusOptions
from documents.utils import copy_basic_file_stats
from documents.utils import copy_file_with_basic_stats
from documents.utils import maybe_override_pixel_limit
from paperless.config import BarcodeConfig

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("paperless.barcodes")


@dataclass(frozen=True)
class Barcode:
    """
    Holds the information about a single barcode and its location in a document
    """

    page: int
    value: str
    settings: BarcodeConfig

    @property
    def is_separator(self) -> bool:
        """
        Returns True if the barcode value equals the configured separation value,
        False otherwise
        """
        return self.value == self.settings.barcode_string

    @property
    def is_asn(self) -> bool:
        """
        Returns True if the barcode value matches the configured ASN prefix,
        False otherwise
        """
        return self.value.startswith(self.settings.barcode_asn_prefix)

    @property
    def is_tag(self) -> bool:
        """
        Returns True if the barcode value matches any configured tag mapping pattern,
        False otherwise.

        Note: This does NOT exclude ASN or separator barcodes - they can also be used
        as tags if they match a tag mapping pattern (e.g., {"ASN12.*": "JOHN"}).
        """
        for regex in self.settings.barcode_tag_mapping:
            if re.match(regex, self.value, flags=re.IGNORECASE):
                return True
        return False


class BarcodePlugin(ConsumeTaskPlugin):
    NAME: str = "BarcodePlugin"

    @property
    def able_to_run(self) -> bool:
        """
        Able to run if:
          - ASN from barcode detection is enabled or
          - Barcode support is enabled and the mime type is supported
        """
        if self.settings.barcode_enable_tiff_support:
            supported_mimes: set[str] = {"application/pdf", "image/tiff"}
        else:
            supported_mimes = {"application/pdf"}

        return (
            self.settings.barcode_enable_asn
            or self.settings.barcodes_enabled
            or self.settings.barcode_enable_tag
            or self.settings.barcode_enable_metadata
        ) and self.input_doc.mime_type in supported_mimes

    def get_settings(self) -> BarcodeConfig:
        """
        Returns the settings for this plugin (Django settings or app config)
        """
        return BarcodeConfig()

    def __init__(
        self,
        input_doc: ConsumableDocument,
        metadata: DocumentMetadataOverrides,
        status_mgr: ProgressManager,
        base_tmp_dir: Path,
        task_id: str,
    ) -> None:
        super().__init__(
            input_doc,
            metadata,
            status_mgr,
            base_tmp_dir,
            task_id,
        )
        # need these for able_to_run
        self.settings = self.get_settings()

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(
            dir=self.base_tmp_dir,
            prefix="barcode",
        )
        self.pdf_file: Path = self.input_doc.original_file
        self._tiff_conversion_done = False
        self.barcodes: list[Barcode] = []

    def _apply_detected_asn(self, detected_asn: int) -> None:
        """
        Apply a detected ASN to metadata if allowed.
        """
        if (
            self.metadata.skip_asn_if_exists
            and Document.global_objects.filter(
                archive_serial_number=detected_asn,
            ).exists()
        ):
            logger.info(
                f"Found ASN in barcode {detected_asn} but skipping because it already exists.",
            )
            return

        logger.info(f"Found ASN in barcode: {detected_asn}")
        self.metadata.asn = detected_asn

    def run(self) -> None:
        # Some operations may use PIL, override pixel setting if needed
        maybe_override_pixel_limit()

        # Maybe do the conversion of TIFF to PDF
        self.convert_from_tiff_to_pdf()

        # Locate any barcodes in the files
        self.detect()

        # try reading tags from barcodes
        # If tag splitting is enabled, skip this on the original document - let each split document extract its own tags
        # However, if we're processing a split document (original_path is set), extract tags
        if (
            self.settings.barcode_enable_tag
            and (
                not self.settings.barcode_tag_split
                or self.input_doc.original_path is not None
            )
            and (tags := self.tags) is not None
            and len(tags) > 0
        ):
            if self.metadata.tag_ids:
                self.metadata.tag_ids += tags
            else:
                self.metadata.tag_ids = tags
            logger.info(f"Found tags in barcode: {tags}")

        # try reading metadata from barcodes
        if self.settings.barcode_enable_metadata and (
            overrides := self.metadata_overrides
        ):
            self.metadata.update(overrides)
            logger.info("Found metadata in barcode")

        # Lastly attempt to split documents
        if self.settings.barcodes_enabled and (
            separator_pages := self.get_separation_pages()
        ):
            # We have pages to split against

            # Note this does NOT use the base_temp_dir, as that will be removed
            tmp_dir = Path(
                tempfile.mkdtemp(
                    dir=settings.SCRATCH_DIR,
                    prefix="paperless-barcode-split-",
                ),
            ).resolve()

            from documents import tasks

            # Create the split document tasks
            for new_document in self.separate_pages(separator_pages):
                copy_file_with_basic_stats(new_document, tmp_dir / new_document.name)

                task = tasks.consume_file.delay(
                    ConsumableDocument(
                        # Same source, for templates
                        source=self.input_doc.source,
                        mailrule_id=self.input_doc.mailrule_id,
                        # Can't use same folder or the consume might grab it again
                        original_file=(tmp_dir / new_document.name).resolve(),
                        # Adding optional original_path for later uses in
                        # workflow matching
                        original_path=self.input_doc.original_file,
                    ),
                    # All the same metadata
                    self.metadata,
                )
                logger.info(f"Created new task {task.id} for {new_document.name}")

            # This file is now two or more files
            self.input_doc.original_file.unlink()

            msg = "Barcode splitting complete!"

            # Update the progress to complete
            self.status_mgr.send_progress(ProgressStatusOptions.SUCCESS, msg, 100, 100)

            # Request the consume task stops
            raise StopConsumeTaskError(msg)

        # Update/overwrite an ASN if possible
        # After splitting, as otherwise each split document gets the same ASN
        if self.settings.barcode_enable_asn and (located_asn := self.asn) is not None:
            self._apply_detected_asn(located_asn)

    def cleanup(self) -> None:
        self.temp_dir.cleanup()

    def convert_from_tiff_to_pdf(self) -> None:
        """
        May convert a TIFF image into a PDF, if the input is a TIFF and
        the TIFF has not been made into a PDF
        """
        # Nothing to do, pdf_file is already assigned correctly
        if self.input_doc.mime_type != "image/tiff" or self._tiff_conversion_done:
            return

        self.pdf_file = convert_from_tiff_to_pdf(
            self.input_doc.original_file,
            Path(self.temp_dir.name),
        )
        self._tiff_conversion_done = True

    @staticmethod
    def read_barcodes_zxing(image: Image.Image) -> list[str]:
        barcodes = []

        import zxingcpp

        detected_barcodes = zxingcpp.read_barcodes(image)
        for barcode in detected_barcodes:
            if barcode.text:
                barcodes.append(barcode.text)
                logger.debug(
                    f"Barcode of type {barcode.format} found: {barcode.text}",
                )

        return barcodes

    def detect(self) -> None:
        """
        Scan all pages of the PDF as images, updating barcodes and the pages
        found on as we go
        """
        # Bail if barcodes already exist
        if self.barcodes:
            return

        # No op if not a TIFF
        self.convert_from_tiff_to_pdf()

        try:
            # Read number of pages from pdf
            with Pdf.open(self.pdf_file) as pdf:
                num_of_pages = len(pdf.pages)
            logger.debug(f"PDF has {num_of_pages} pages")

            # Get limit from configuration
            barcode_max_pages: int = (
                num_of_pages
                if self.settings.barcode_max_pages == 0
                else self.settings.barcode_max_pages
            )

            if barcode_max_pages < num_of_pages:  # pragma: no cover
                logger.debug(
                    f"Barcodes detection will be limited to the first {barcode_max_pages} pages",
                )

            # Loop al page
            for current_page_number in range(min(num_of_pages, barcode_max_pages)):
                logger.debug(f"Processing page {current_page_number}")

                # Convert page to image
                page = convert_from_path(
                    self.pdf_file,
                    dpi=self.settings.barcode_dpi,
                    output_folder=self.temp_dir.name,
                    first_page=current_page_number + 1,
                    last_page=current_page_number + 1,
                )[0]

                # Remember filename, since it is lost by upscaling
                page_filepath = Path(page.filename)
                logger.debug(f"Image is at {page_filepath}")

                # Upscale image if configured
                factor = self.settings.barcode_upscale
                if factor > 1.0:
                    logger.debug(
                        f"Upscaling image by {factor} for better barcode detection",
                    )
                    x, y = page.size
                    page = page.resize(
                        (round(x * factor), (round(y * factor))),
                    )

                # Detect barcodes
                for barcode_value in self.read_barcodes_zxing(page):
                    self.barcodes.append(
                        Barcode(current_page_number, barcode_value, self.settings),
                    )

                # Delete temporary image file
                page_filepath.unlink()

        # Password protected files can't be checked
        # This is the exception raised for those
        except PasswordError as e:
            logger.warning(
                f"File is likely password protected, not checking for barcodes: {e}",
            )
        # This file is really borked, allow the consumption to continue
        # but it may fail further on
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Exception during barcode scanning: {e}",
            )

    @property
    def asn(self) -> int | None:
        """
        Search the parsed barcodes for any ASNs.
        The first barcode that starts with barcode_asn_prefix
        is considered the ASN to be used.
        Returns the detected ASN (or None)
        """
        asn = None

        # Ensure the barcodes have been read
        self.detect()

        # get the first barcode that starts with barcode_asn_prefix
        asn_text: str | None = next(
            (x.value for x in self.barcodes if x.is_asn),
            None,
        )

        if asn_text:
            logger.debug(f"Found ASN Barcode: {asn_text}")
            # remove the prefix and remove whitespace
            asn_text = asn_text[len(self.settings.barcode_asn_prefix) :].strip()

            # remove non-numeric parts of the remaining string
            asn_text = re.sub(r"\D", "", asn_text)

            # now, try parsing the ASN number
            try:
                asn = int(asn_text)
            except ValueError as e:
                logger.warning(f"Failed to parse ASN number because: {e}")

        return asn

    @property
    def tags(self) -> list[int]:
        """
        Search the parsed barcodes for any tags.
        Returns the detected tag ids (or empty list)
        """
        tags: list[int] = []

        # Ensure the barcodes have been read
        self.detect()

        for x in self.barcodes:
            tag_texts: str = x.value

            for raw in tag_texts.split(","):
                try:
                    tag_str: str | None = None
                    for regex in self.settings.barcode_tag_mapping:
                        if re.match(regex, raw, flags=re.IGNORECASE):
                            sub = self.settings.barcode_tag_mapping[regex]
                            tag_str = (
                                re.sub(regex, sub, raw, flags=re.IGNORECASE)
                                if sub
                                else raw
                            )
                            break

                    if tag_str:
                        tag, _ = Tag.objects.get_or_create(
                            name__iexact=tag_str,
                            defaults={"name": tag_str},
                        )

                        logger.debug(
                            f"Found Tag Barcode '{raw}', substituted "
                            f"to '{tag}' and mapped to "
                            f"tag #{tag.pk}.",
                        )
                        tags.append(tag.pk)

                except Exception as e:
                    logger.error(
                        f"Failed to find or create TAG '{raw}' because: {e}",
                    )

        return tags

    @property
    def metadata_overrides(self) -> DocumentMetadataOverrides | None:
        """
        Extract document metadata from barcodes using configurable regex patterns.
        Supports named groups for: correspondent, document_type, tags,
        title, owner, created, and custom_field_name/custom_field_value.
        Tags can be comma-separated.
        Regex substitution is applied like tag barcode mapping.
        If only numbered groups are present, the first two groups are treated as
        custom field name/value.
        """
        if not self.settings.barcode_enable_metadata:
            return None

        if not self.settings.barcode_metadata_mapping:
            return None

        # Ensure the barcodes have been read
        self.detect()

        overrides = DocumentMetadataOverrides()
        custom_fields: dict[int, str] = {}
        seen_custom_field_keys: set[str] = set()
        auto_create = {
            value.lower()
            for value in (self.settings.barcode_metadata_auto_create or [])
        }

        def _apply_substitution(
            raw: str,
            pattern: str,
            substitution: str,
        ) -> str | None:
            """Apply regex substitution to extract value from barcode text."""
            if not re.match(pattern, raw, flags=re.IGNORECASE):
                return None
            return (
                re.sub(pattern, substitution, raw, flags=re.IGNORECASE)
                if substitution
                else raw
            )

        def _extract_pair(match: re.Match) -> tuple[str | None, str | None]:
            gd = match.groupdict()
            # Prefer explicit name/value pairs
            if "custom_field_name" in gd:
                name = gd.get("custom_field_name")
                value = gd.get("custom_field_value")
                return name, value
            # Fallback to numbered groups
            groups = match.groups()
            if len(groups) >= 2:
                return groups[0], groups[1]
            return None, None

        for barcode in self.barcodes:
            text = barcode.value
            for pattern, substitution in self.settings.barcode_metadata_mapping.items():
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue

                gd = match.groupdict()

                # Correspondent
                if "correspondent" in gd and gd.get("correspondent"):
                    correspondent_name = _apply_substitution(
                        text,
                        pattern,
                        substitution,
                    )
                    if correspondent_name and overrides.correspondent_id is None:
                        try:
                            from documents.models import Correspondent

                            if "correspondent" in auto_create:
                                correspondent, created = (
                                    Correspondent.objects.get_or_create(
                                        name__iexact=correspondent_name,
                                        defaults={"name": correspondent_name},
                                    )
                                )
                                if created:
                                    logger.info(
                                        f"Auto-created correspondent '{correspondent_name}' (id={correspondent.pk})",
                                    )
                            else:
                                correspondent = Correspondent.objects.get(
                                    name__iexact=correspondent_name,
                                )
                            overrides.correspondent_id = correspondent.pk
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve correspondent '{correspondent_name}': {e}",
                            )

                # Document type
                if "document_type" in gd and gd.get("document_type"):
                    doc_type_name = _apply_substitution(text, pattern, substitution)
                    if doc_type_name and overrides.document_type_id is None:
                        try:
                            from documents.models import DocumentType

                            if "document_type" in auto_create:
                                doc_type, _ = DocumentType.objects.get_or_create(
                                    name__iexact=doc_type_name,
                                    defaults={"name": doc_type_name},
                                )
                            else:
                                doc_type = DocumentType.objects.get(
                                    name__iexact=doc_type_name,
                                )
                            overrides.document_type_id = doc_type.pk
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve document type '{doc_type_name}': {e}",
                            )

                # Tags
                if ("tag" in gd and gd.get("tag")) or ("tags" in gd and gd.get("tags")):
                    tags_value = _apply_substitution(text, pattern, substitution)
                    if tags_value:
                        tag_names = [
                            name.strip()
                            for name in tags_value.split(",")
                            if name.strip()
                        ]
                        for tag_name in tag_names:
                            try:
                                from documents.models import Tag

                                if "tag" in auto_create or "tags" in auto_create:
                                    tag, created = Tag.objects.get_or_create(
                                        name__iexact=tag_name,
                                        defaults={"name": tag_name},
                                    )
                                    if created:
                                        logger.info(
                                            f"Auto-created tag '{tag_name}' (id={tag.pk})",
                                        )
                                else:
                                    tag = Tag.objects.get(name__iexact=tag_name)

                                if overrides.tag_ids is None:
                                    overrides.tag_ids = []
                                if tag.pk not in overrides.tag_ids:
                                    overrides.tag_ids.append(tag.pk)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to resolve tag '{tag_name}': {e}",
                                )

                # Title
                if "title" in gd and gd.get("title"):
                    title_value = _apply_substitution(text, pattern, substitution)
                    if title_value and overrides.title is None:
                        overrides.title = title_value

                # Owner
                if "owner" in gd and gd.get("owner"):
                    owner_value = _apply_substitution(text, pattern, substitution)
                    if owner_value and overrides.owner_id is None:
                        try:
                            from django.contrib.auth import get_user_model

                            User = get_user_model()
                            owner = User.objects.get(username__iexact=owner_value)
                            overrides.owner_id = owner.pk
                        except Exception as e:
                            logger.warning(
                                f"Failed to resolve owner '{owner_value}': {e}",
                            )

                # Created date
                if "created" in gd and gd.get("created"):
                    created_value = _apply_substitution(text, pattern, substitution)
                    if created_value and overrides.created is None:
                        try:
                            overrides.created = datetime.date.fromisoformat(
                                created_value,
                            )
                        except Exception:
                            logger.warning(
                                f"Failed to parse created date '{created_value}'",
                            )

                # Custom fields
                if "custom_field_name" in gd or len(match.groups()) >= 2:
                    cf_result = _apply_substitution(text, pattern, substitution)
                    if cf_result:
                        # Parse the result as "name=value"
                        if "=" in cf_result:
                            cf_name, cf_value = cf_result.split("=", 1)
                        else:
                            # Fallback to direct extraction if no = in result
                            cf_name, cf_value = _extract_pair(match)

                        if cf_name and cf_value:
                            if cf_name in seen_custom_field_keys:
                                logger.warning(
                                    f"Custom field '{cf_name}' already set, ignoring '{cf_value}'",
                                )
                            else:
                                try:
                                    from documents.models import CustomField

                                    if "custom_field" in auto_create:
                                        cf_obj, created = (
                                            CustomField.objects.get_or_create(
                                                name=cf_name,
                                                defaults={
                                                    "data_type": CustomField.FieldDataType.STRING,
                                                },
                                            )
                                        )
                                        if created:
                                            logger.info(
                                                f"Auto-created custom field '{cf_name}' (id={cf_obj.id})",
                                            )
                                    else:
                                        cf_obj = CustomField.objects.get(name=cf_name)

                                    custom_fields[cf_obj.id] = cf_value
                                    seen_custom_field_keys.add(cf_name)
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to resolve custom field '{cf_name}': {e}",
                                    )

        if custom_fields:
            overrides.custom_fields = custom_fields

        has_overrides = any(
            value is not None
            for value in (
                overrides.title,
                overrides.correspondent_id,
                overrides.document_type_id,
                overrides.tag_ids,
                overrides.created,
                overrides.owner_id,
                overrides.custom_fields,
            )
        )

        return overrides if has_overrides else None

    def get_separation_pages(self) -> dict[int, bool]:
        """
        Search the parsed barcodes for separators and returns a dict of page
        numbers, which separate the file into new files, together with the
        information whether to keep the page.
        """
        # filter all barcodes for the separator string
        # get the page numbers of the separating barcodes
        retain = self.settings.barcode_retain_split_pages
        separator_pages = {
            bc.page: retain
            for bc in self.barcodes
            if bc.is_separator and (not retain or (retain and bc.page > 0))
        }  # as below, dont include the first page if retain is enabled

        # add the page numbers of the ASN barcodes
        # (except for first page, that might lead to infinite loops).
        if self.settings.barcode_enable_asn:
            separator_pages = {
                **separator_pages,
                **{bc.page: True for bc in self.barcodes if bc.is_asn and bc.page != 0},
            }

        # add the page numbers of the TAG barcodes if splitting is enabled
        # (except for first page, that might lead to infinite loops).
        if self.settings.barcode_tag_split and self.settings.barcode_enable_tag:
            separator_pages = {
                **separator_pages,
                **{bc.page: True for bc in self.barcodes if bc.is_tag and bc.page != 0},
            }

        return separator_pages

    def separate_pages(self, pages_to_split_on: dict[int, bool]) -> list[Path]:
        """
        Separate the provided pdf file on the pages_to_split_on.
        The pages which are defined by the keys in page_numbers
        will be removed if the corresponding value is false.
        Returns a list of (temporary) filepaths to consume.
        These will need to be deleted later.
        """

        document_paths = []
        fname: str = self.input_doc.original_file.stem
        with Pdf.open(self.pdf_file) as input_pdf:
            # Start with an empty document
            current_document: list[Page] = []
            # A list of documents, ie a list of lists of pages
            documents: list[list[Page]] = [current_document]

            for idx, page in enumerate(input_pdf.pages):
                # Keep building the new PDF as long as it is not a
                # separator index
                if idx not in pages_to_split_on:
                    current_document.append(page)
                    continue

                # This is a split index
                # Start a new destination page listing
                logger.debug(f"Starting new document at idx {idx}")
                current_document = []
                documents.append(current_document)
                keep_page: bool = pages_to_split_on[idx]
                if keep_page:
                    # Keep the page
                    # (new document is started by asn barcode)
                    current_document.append(page)

            documents = [x for x in documents if len(x)]

            logger.debug(f"Split into {len(documents)} new documents")

            # Write the new documents out
            for doc_idx, document in enumerate(documents):
                dst = Pdf.new()
                dst.pages.extend(document)

                output_filename = f"{fname}_document_{doc_idx}.pdf"

                logger.debug(f"pdf no:{doc_idx} has {len(dst.pages)} pages")
                savepath = Path(self.temp_dir.name) / output_filename
                with savepath.open("wb") as out:
                    dst.save(out)

                copy_basic_file_stats(self.input_doc.original_file, savepath)

                document_paths.append(savepath)

            return document_paths
