import logging
from types import NoneType
from typing import Optional, Any, Type, List, Union, get_origin, get_args, Tuple
from os import environ

logger = logging.getLogger(__name__)


def __get_public_attrs(clz) -> List[str]:
    return [attribute for attribute in vars(clz) if "__" not in attribute]


def __env_var_name_of_attr(clazz: Type, attr_name: str, prefix: str) -> str:
    return f"{prefix}{clazz.__name__}_{attr_name}".upper()


def __load_from_envs(env_prefix: str = None):
    if env_prefix is None:
        env_prefix = ""

    def cast(settings_class, attr_name, attr_str_val) -> Any:
        annotations = getattr(settings_class, "__annotations__")
        if attr_name not in annotations:
            return attr_str_val

        declared_type = annotations[attr_name]
        if get_origin(declared_type) is Union:
            union_types: Tuple[Any] = get_args(declared_type)
            if len(union_types) == 2:
                if union_types[0] is NoneType:
                    declared_type = union_types[1]
                elif union_types[1] is NoneType:
                    declared_type = union_types[0]
                else:
                    declared_type = None
            else:
                declared_type = None

        if declared_type is None:
            return attr_str_val

        try:
            return declared_type(attr_str_val)
        except Exception as ex:
            print(f"Cannot cast custom value of {attr_name} of class {settings_class}. Error: {ex}")
        return attr_str_val

    def lookup_env_configuration(settings_class, attr_name: str, prefix: str) -> Optional[str]:
        env_name = __env_var_name_of_attr(settings_class, attr_name, prefix)
        if env_name in environ:
            # logger.warning("Reading custom setting from env: %s", env_name)
            return environ[env_name]
        else:
            # logger.warning("No custom setting from env: " + env_name)
            return None

    def decorator(settings_class, prefix=env_prefix):
        attribute_list: List[str] = __get_public_attrs(settings_class)
        for attr_name in attribute_list:
            configuration = lookup_env_configuration(settings_class, attr_name, prefix)
            if configuration is None:
                default_value = settings_class.__getattribute__(settings_class, attr_name)
                if isinstance(default_value, __RequiredSetting):
                    default_value.setting_name = __env_var_name_of_attr(settings_class, attr_name, prefix)  # set variable name
                    default_value.setting_was_not_set()
                continue
            config_value = cast(settings_class, attr_name, configuration)
            setattr(settings_class, attr_name, config_value)
        return settings_class

    return decorator


class __RequiredSetting:
    def __init__(self, setting_description: Optional[str] = None, fail_fast: bool = False):
        self.__setting_description: str = setting_description
        self.setting_name: Optional[str] = None
        self.__fail_fast = fail_fast

    def __set_name__(self, owner, name):
        self.setting_name = name

    def __get_error_str(self):
        err_str = f"Required setting was not found in environment variable: {self.setting_name}"
        if self.__setting_description is not None:
            err_str += f"\nSetting is used for: {self.__setting_description}"

        return err_str

    def setting_was_not_set(self):
        if self.__fail_fast:
            raise RuntimeError(self.__get_error_str())

    def __get__(self, obj, objtype=None):
        raise RuntimeError(self.__get_error_str())
