"""
"""
import logging
from formencode import Schema, Invalid
from formencode.api import FancyValidator
from formencode.validators import FormValidator, Int, UnicodeString, Empty, NotEmpty, ConfirmType, Set, OneOf, Bool
from formencode.compound import Pipe, Any as AnyMatch
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Any, Optional, get_origin, get_args, Annotated, Union
from types import UnionType, NoneType
from inspect import get_annotations
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@dataclass
class Profile:
    image_url: str
    width: int
    height: int

@dataclass
class User:
     """A new type describing a User"""
     name: Annotated[str, 15]
     profile: Profile | None
     groups: set[str] = field(default_factory=set)
     email: str | None = None
     verified: Any = (1, 2)
     username: str = ""
     num_logins: int = 0
     is_person: bool = False
     


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

NOT_SET = object()
PLACEHOLDER = object()

@dataclass
class OurMetadata:
    not_empty: bool|None = None
    if_empty: Any|None = NOT_SET
    if_missing: Any|None = NOT_SET

@dataclass
class SchemaBuilder:
    """
    This takes a root object and builds a schema to validate
    and coerce the output a json.loads-type structure into the root object
    and all nested objects (all the way down, turtles).
    """
    def merge_metadata(self, items):
        #raise Warning('This is not complete')
        return items[0] if items else None

    def get_validator(self, annotation, field_obj=NOT_SET, part_of_field_obj=NOT_SET, metadata=None):
        """ Resolve a validator for the given annotation. """
        a_args = get_args(annotation)
        if get_origin(annotation) is Annotated:
            # Unwrap annotation and keep metadata.
            source_type = a_args[0]
            a_args = get_args(annotation)
            metadata = self.merge_metadata([m for m in annotation.__metadata__ if isinstance(m, OurMetadata)])
        else:
            source_type = annotation
        origin = get_origin(source_type)
        # Plain types don't return an origin.
        if origin is None:
            origin = source_type

        # Normalize this for Optional or Union.
        if origin is Union:
            origin = UnionType

        default_set = field_obj is not NOT_SET and (field_obj.default is not MISSING or field_obj.default_factory is not MISSING)
        v = None
        v_args = []
        v_kwargs = {}

        if issubclass(origin, NoneType):
            v = ConfirmType
            v_kwargs['subclass'] = NoneType
        elif issubclass(origin, UnionType) :
            # This handles
            # - Optional[t]
            # - Union[t1, ...]
            # - t1 | t2 | ...
            # - t | None
            # Note that nested unions, including a union inside Optional are
            # flattened by get_args / get_origin.
            v = AnyMatch
            # There should realistically be only one of these as I understand it.
            none_types = [t for t in a_args if issubclass(t, NoneType)]
            v_args.extend([self.get_validator(t, part_of_field_obj=field_obj) for t in a_args if not issubclass(t, NoneType)])
            if none_types:
                v_args.append(self.get_validator(none_types[0], part_of_field_obj=field_obj))
        elif issubclass(origin, Any):
            # @TODO: Is there a better way to handle this?
            v = PassThrough
        elif issubclass(origin, str):
            v = UnicodeString
        elif issubclass(origin, bool):
            # This must come before int.
            v = Bool
        elif issubclass(origin, int):
            v = Int
        elif issubclass(origin, set):
            v = Set
            v_kwargs['use_set'] = True
            # @TODO: When should we recurse?
        elif self.is_namespace(origin):
            v = self.build
            v_args.append(origin)
        if v is None:
            raise InvalidTarget(f'Unsupported type {origin}')
        if default_set:
            # @TODO: Or can just inject a value to remove later.
            #default = field_obj.default if field_obj.default else field_obj.default_factory()
            v_kwargs['if_missing'] = PLACEHOLDER
            v_kwargs['if_empty'] = PLACEHOLDER
        elif v == self.build or (not issubclass(v, AnyMatch) and not issubclass(v, PassThrough)):
            v_kwargs['not_empty'] = True
        logger.info(f'Resolved {field_obj.name if field_obj is not NOT_SET else None}:{annotation}:{origin}:{a_args} to {v}(*{v_args}, **{v_kwargs})')
        return v(*v_args, **v_kwargs)

    def is_namespace(self, obj):
        """ Is this a namespace-like structure we can """
        return is_dataclass(obj)

    def get_field_map(self, target):
        return {f.name: f for f in fields(target)}

    def build(self, target, if_missing=NOT_SET, if_empty=NOT_SET, not_empty=NOT_SET):
        if not self.is_namespace(target):
            raise InvalidTarget(f'Unsupported target {target}')
        field_map = self.get_field_map(target)
        annotations = get_annotations(target)
        kwargs = {field_name: self.get_validator(annotation, field_obj=field_map.get(field_name, NOT_SET)) for field_name, annotation in annotations.items()}
        if if_missing is not NOT_SET:
            kwargs['if_missing'] = if_missing
        if if_empty is not NOT_SET:
            kwargs['if_empty'] = if_empty
        if not_empty is not NOT_SET:
            kwargs['not_empty'] = not_empty
        return Pipe(BaseSchema(**kwargs), DataclassFactory(target=target))


class DataclassFactory(FancyValidator):
    accept_iterator = True
    # ? not sure what this should be
    #compound = True
    __unpackargs__ = ['target']

    def _convert_to_python(self, value, state):
        # Remove placeholder we injected earlier so the dataclass can use its own default mechanism.
        return self.target(**{k: v for (k, v) in value.items() if v is not PLACEHOLDER})


class PassThrough(FancyValidator):
    def _convert_to_python(self, value, state):
        return value

def main():
    msg = {
         "name":"alice",
         "groups": [
              "admin",
              "engineering"
         ],
         "email":None,
         "is_person": 7,
        "profile": {
            "image_url": "/mugshot.png",
            "width": 100,
            "height": 100,
           }

    }
    builder = SchemaBuilder()
    schema = builder.build(User)
    from pprint import pprint
    pprint(schema)# (f'built schema {schema}')
    pprint (schema.to_python(msg))


@dataclass
class Num2:
    num2: int
    num3: int

@dataclass
class Num1:
    num1: int
    nested: Num2|None

if __name__ == '__main__':
    main()

    # What if we just rewrote formencode validators to use dataclasses or structs?
