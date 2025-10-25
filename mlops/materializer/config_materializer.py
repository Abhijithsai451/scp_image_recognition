import json
from typing import Type, Any, Optional
from pathlib import Path

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from parse_config import ConfigParser

CONFIG_FILENAME = "config.json"
RESUME_FILENAME = "resume_path.txt"


class ConfigParserMaterializer(BaseMaterializer):
    """
    Materializer for Configparser objects.
    """
    ASSOCIATED_TYPES = (ConfigParser,)
    ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> ConfigParser:
        """
        Reads a ConfigParser object from a json file.
        Args:
            data_type: The type of the data to read.
        Returns:
            Loaded ConfigParser object.
        """
        # Call super().load() to ensure artifact context is set up by ZenML
        super().load(data_type)

        # Use self.uri for the artifact path
        config_file_path = Path(self.uri) / CONFIG_FILENAME

        # Load the raw config dictionary
        with fileio.open(str(config_file_path), "r") as f:
            config_data = json.loads(f.read())

        resume_file_path = Path(self.uri) / RESUME_FILENAME
        resume_path: Optional[Path] = None
        if fileio.exists(str(resume_file_path)):
            with fileio.open(str(resume_file_path), "r") as f:
                resume_str = f.read()
                if resume_str:
                    resume_path = Path(resume_str)

        # Use the special _reconstruct method to avoid side-effects
        return ConfigParser._reconstruct(config_dict=config_data, resume_path=resume_path)

    def save(self, config_parser_instance: ConfigParser) -> None:
        """Writes a ConfigParser object to a JSON file.

        Args:
            config_parser_instance: The ConfigParser object to write.
        """
        # Call super().save() to ensure artifact context is set up by ZenML
        super().save(config_parser_instance)

        # Use self.uri for the artifact path
        config_file_path = Path(self.uri) / CONFIG_FILENAME

        # Save the internal config dictionary
        with fileio.open(str(config_file_path), "w") as f:
            json.dump(config_parser_instance.config, f, indent=4)  # Use .config property

        # Save the resume path if it exists
        if config_parser_instance.resume:
            resume_file_path = Path(self.uri) / RESUME_FILENAME
            with fileio.open(str(resume_file_path), "w") as f:
                f.write(str(config_parser_instance.resume))