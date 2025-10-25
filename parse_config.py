import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from pydantic import BaseModel, PrivateAttr, ConfigDict
from typing import Dict, Any, Optional


# Helper functions (as they were)
def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    return reduce(getitem, keys, tree)

def setup_logging(log_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=log_dir / 'app.log')
    print(f"Logging set up at {log_dir / 'app.log'}")


def read_json(filepath):
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def write_json(data, filepath):
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON written to {filepath}")


class ConfigParser(BaseModel):
    # Pydantic configuration: allow arbitrary types to be passed through.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Store the actual configuration dictionary directly.
    _config_data: Dict[str, Any] = PrivateAttr()
    _resume_path: Optional[Path] = PrivateAttr()

    # These will be initialized *once* during initial creation or lazily within a step
    _save_dir_path: Optional[Path] = PrivateAttr(default=None)
    _log_dir_path: Optional[Path] = PrivateAttr(default=None)
    _log_levels_map: Dict[int, int] = PrivateAttr(default_factory=dict)

    def __init__(self, config_dict:Optional[Dict[str, Any]], resume_path: Optional[Path] = None, run_id: Optional[str] = None,
                 **data):
        super().__init__(**data)
        self._config_data = config_dict
        self._resume_path = resume_path
        # Only perform side-effects if we are the "initial" constructor call.
        # This will be run in main.py, not during ZenML deserialization.
        self._initialize_paths_and_logging(run_id)

    def _initialize_paths_and_logging(self, run_id: Optional[str] = None):
        """Initializes save_dir, log_dir and logging, intended to be run once."""
        save_dir_base = Path(self._config_data['trainer']['save_dir'])
        exper_name = self._config_data['name']
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir_path = save_dir_base / 'models' / exper_name / run_id
        self._log_dir_path = save_dir_base / 'log' / exper_name / run_id

        self._save_dir_path.mkdir(parents=True, exist_ok=True)
        self._log_dir_path.mkdir(parents=True, exist_ok=True)
        write_json(self._config_data, self._save_dir_path / 'config.json')
        setup_logging(self._log_dir_path)

        self._log_levels_map = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        resume_path = None
        if args.resume is not None:
            resume_path = Path(args.resume)
            cfg_fname = resume_path.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            cfg_fname = Path(args.config)

        config_dict_from_file = read_json(cfg_fname)
        if args.config and resume_path:
            config_dict_from_file.update(read_json(args.config))

        # Apply modifications directly to the config_dict_from_file
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags))
                        for opt in options if getattr(args, _get_opt_name(opt.flags)) is not None}
        updated_config_dict = _update_config(config_dict_from_file, modification)

        # Now, instantiate ConfigParser with the final dictionary and resume path.
        return cls(config_dict=updated_config_dict, resume_path=resume_path)

    @classmethod
    def _reconstruct(cls, config_dict: Dict[str, Any], resume_path: Optional[Path] = None):
        """
        Internal factory method for ZenML Materializer to reconstruct ConfigParser.
        This constructor avoids side-effects like creating directories or setting up logging.
        """
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        super(ConfigParser, instance).__init__()  # Initialize BaseModel part
        instance._config_data = config_dict
        instance._resume_path = resume_path
        # Paths and logging are not initialized here; they can be lazily initialized via properties if accessed.
        instance._save_dir_path = None
        instance._log_dir_path = None
        instance._log_levels_map = {  # Default log levels if _initialize_paths_and_logging is not called
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        return instance

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self._config_data[name]['type']
        module_args = dict(self._config_data[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self._config_data[name]['type']
        module_args = dict(self._config_data[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        return self._config_data[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self._log_levels_map.keys())
        assert verbosity in self._log_levels_map, msg_verbosity
        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(self._log_levels_map[verbosity])
        return logger_instance

    @property
    def config(self):
        return self._config_data

    @property
    def save_dir(self):
        # Lazily initialize if not already done, useful if _reconstruct was used
        if self._save_dir_path is None:
            self._initialize_paths_and_logging()
        return self._save_dir_path

    @property
    def log_dir(self):
        # Lazily initialize if not already done
        if self._log_dir_path is None:
            self._initialize_paths_and_logging()
        return self._log_dir_path

    @property
    def resume(self):
        return self._resume_path