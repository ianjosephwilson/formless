import logging
from formencode import Schema, Invalid
from formencode.api import FancyValidator
from formencode.validators import FormValidator, Int, UnicodeString, Empty, NotEmpty, ConfirmType, Set, OneOf
from formencode.compound import Pipe, Any
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import get_type_hints, Any, Optional, get_origin, get_args
from types import UnionType, NoneType
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



@dataclass
class User:
     """A new type describing a User"""
     name: str
     groups: set[str] = field(default_factory=set)
     email: str | None = None
     username: Optional[str] = ""
     num_logins: int = 0


class BaseSchema(Schema):
    """Schema with sane settings."""

    allow_extra_fields = True
    filter_extra_fields = True


def get_invalid_from_errors(errors, field_dict, state):
    """Roll dictionary of errors into an Invalid exception and return it."""
    error_list = list(errors.items())
    error_list.sort()
    error_message = "<br>\n".join(
        ["%s: %s" % (name, value) for name, value in error_list]
    )
    return Invalid(error_message, field_dict, state, error_dict=errors)


class BaseFormValidator(FormValidator):
    """FormValidator with utility method."""

    def errors_to_invalid(self, errors, field_dict, state):
        """Roll a dictionary of errors into Invalid exception and raise it."""
        raise get_invalid_from_errors(errors, field_dict, state)


class InvalidTarget(Exception):
    pass

def index_or_default(items, item, default=None):
     try:
          return index(items)
     except IndexError:
          return default

NotSet = object()

@dataclass
class Config:

     def is_supported_root(self, root):
          return is_dataclass(root)

     def get_validator(self, v, field_obj=None):
          origin = get_origin(v)
          if origin is None:
               origin = v
          args = get_args(v)
          default_set = field_obj and field_obj.default is not MISSING
          v = None
          v_args = []
          v_kwargs = {}
          import typing
          import types
          # @TODO: This is weird: Union versus UnionType, trying to add Optional support.
          if origin != typing.Union and issubclass(origin, NoneType):
               v = ConfirmType
               v_kwargs['issubclass'] = NoneType
          # @TODO: This is also weird: Union versus UnionType, trying to add Optional support.
          elif origin == typing.Union or issubclass(origin, UnionType) :
               v = OneOf
               v_args = [[self.get_validator(t) for t in args]]
               # TODO
               if default_set:
                    v_kwargs['if_missing'] = field_obj.default
                    v_kwargs['if_empty'] = field_obj.default
               else:
                    v_kwargs['not_empty'] = True
          elif issubclass(origin, str):
               v = UnicodeString
          elif issubclass(origin, int):
               kwargs = {}
               if default_set:
                    v_kwargs['if_missing'] = field_obj.default
                    v_kwargs['if_empty'] = field_obj.default
               else:
                    v_kwargs['not_empty'] = True
               v = Int
          elif issubclass(origin, set):
               v = Set
               v_kwargs['use_set'] = True
          elif is_dataclass(origin):
               v = build
               v_args.extend([origin, self])
          if v is None:
               raise InvalidTarget(f'Unsupported type {origin}')
          logger.info(f'Resolved {field_obj.name if field_obj else "No field"}:{origin}:{args} to {v}(*{v_args}, **{v_kwargs})')
          return v(*v_args, **v_kwargs)


def build(target: Any, config: Config) -> Schema:
    if not config.is_supported_root(target):
        raise InvalidTarget(f'Unsupported root {target}')
    field_lookup = {field.name: field for field in fields(target)}
    type_hints = get_type_hints(target, include_extras=True)
    return Pipe(BaseSchema(**{field_name: config.get_validator(type_hint, field_lookup[field_name]) for field_name, type_hint in type_hints.items()}), DataclassFactory(target=target))


class DataclassFactory(FancyValidator):
     accept_iterator = True
     # ? not sure what this should be
     #compound = True
     __unpackargs__ = ['target']

     def _convert_to_python(self, value, state):
          return self.target(**value)


def main():
     msg = {
          "name":"alice",
          "groups": [
               "admin",
               "engineering"
          ],
          "email":None,

     }
     schema = build(User, Config())
     print (schema.to_python(msg))

if __name__ == '__main__':
     main()
