"""
"""
import logging
from formencode import Schema, Invalid
from formencode.api import FancyValidator
from formencode.validators import FormValidator, Int, String, Empty, NotEmpty, ConfirmType, Set, OneOf, Bool, Number
from formencode.compound import Pipe, Any as AnyMatch
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Any, Optional, get_origin, get_args, Annotated, Union
from types import UnionType, NoneType, SimpleNamespace
from inspect import get_annotations
from itertools import chain


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class OurMetadata(SimpleNamespace):
    """ Use in Annotated to pass arguments to formencode. """
    pass


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
     num_logins: Annotated[int, OurMetadata(min=0)] = 0
     is_person: bool = False
     # @TODO: Handle tuples, fixed and variable, mixed: Maybe just make a custom validator to handle this madness.
     empty_tuple: tuple[()]
     tuple_of_any1: tuple
     tuple_of_any2: tuple[Any, ...]
     tuple_of_ints: tuple[int, ...]
     tuple_of_int: tuple[int]
     union_of_tuple_types: tuple[()]|tuple[int]|tuple[int, int]
     # What madman is doing this?
     tuple_of_mixed: tuple[int, str]

     # Maybe a ForEach(Any(Int(), String()))
     mixed_list: list[int|str]

     # I'm not sure formencode supports this.
     mixed_dict: dict[str|int, str|int]

     # @TODO: Handle state based on depth? based on key? based on global state?

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
class VSpec:
    """ Specification to create a validator. """
    # Callable with *args and **kwargs to create validator .
    factory: Any
    # List of args to pass to factory, meant to be mutated.
    args: list|None
    # List of kwargs to pass to factory, meant to be mutated.
    kwargs: dict|None


@dataclass
class VSpecProvider:
    """ Simple validator specification provider. """
    name: str
    factory: Any
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    include_classes: tuple = ()
    exclude_classes: tuple = ()

    def provide_vspec(self, origin, args):
        if issubclass(origin, self.include_classes) and (not self.exclude_classes or not issubclass(origin, self.exclude_classes)):
            return VSpec(
                factory=self.factory,
                args=self.args if self.args is not None else [],
                kwargs=self.kwargs if self.kwargs is not None else {})


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


general_providers = (
    VSpecProvider("int", include_classes=(int,), exclude_classes=(bool,), factory=Int),
    VSpecProvider("float", include_classes=(float,), factory=Number),
    VSpecProvider("str", include_classes=(str,), factory=String),
    VSpecProvider("types.NoneType", include_classes=(NoneType,), factory=ConfirmType, kwargs=dict(subclass=NoneType)),
    VSpecProvider("typing.Any", include_classes=(Any,), factory=PassThrough),
    VSpecProvider("set", include_classes=(set,), factory=Set, kwargs=dict(use_set=True)),
    VSpecProvider("bool", include_classes=(bool,), factory=Bool),
    # @TODO: There is no decimal validator.
    #VSpecProvider("decimal.Decimal", include_classes=(bool,), factory=Decimal),
)


@dataclass
class SchemaBuilder:
    """
    This takes a root object and builds a schema to validate
    and coerce the output a json.loads-type structure into the root object
    and all nested objects (all the way down, turtles).
    """
    debug: bool = False
    provide_for_unions: bool = True
    providers: tuple = general_providers

    def merge_metadatas(self, metadatas):
        return dict(chain(*(m.__dict__.items() for m in metadatas)))

    def get_validator(self, annotation, field_obj=NOT_SET, part_of_field_obj=NOT_SET, metadata=None):
        """ Resolve a validator for the given annotation. """
        a_args = get_args(annotation)
        if get_origin(annotation) is Annotated:
            # Unwrap annotation and keep metadata.
            source_type = a_args[0]
            a_args = get_args(annotation)
            metadata = self.merge_metadatas([m for m in annotation.__metadata__ if isinstance(m, OurMetadata)])
        else:
            metadata = None
            source_type = annotation
        origin = get_origin(source_type)
        # Plain types don't return an origin.
        if origin is None:
            origin = source_type

        # Normalize this for Optional or Union.
        if origin is Union:
            origin = UnionType

        default_set = field_obj is not NOT_SET and (field_obj.default is not MISSING or field_obj.default_factory is not MISSING)

        vspec = None
        # We could probably shoehorn both the union check and namespace check into
        # the provider list but the api would be terrible and I don't think it is
        # worth it at this point.
        if self.provide_for_unions and issubclass(origin, UnionType):
            # This handles
            # - Optional[t]
            # - Union[t1, ...]
            # - t1 | t2 | ...
            # - t | None
            # Note that nested unions, including a union inside Optional are
            # flattened by get_args / get_origin.
            v = AnyMatch
            v_args = []
            # There should realistically be only one of these as I understand it.
            none_types = [t for t in a_args if issubclass(t, NoneType)]
            v_args.extend([self.get_validator(t, part_of_field_obj=field_obj) for t in a_args if not issubclass(t, NoneType)])
            if none_types:
                v_args.append(self.get_validator(none_types[0], part_of_field_obj=field_obj))
            vspec = VSpec(factory=v, args=v_args, kwargs={})
        else:
            for provider in self.providers:
                vspec = provider.provide_vspec(origin, a_args)
                if vspec:
                    break
        if not vspec and self.is_namespace(origin):
            vspec = VSpec(
                factory=self.build,
                args=[origin],
                kwargs={})

        if not vspec:
            raise InvalidTarget(f'Unsupported type {origin}')
        if default_set:
            # @TODO: Or can just inject a value to remove later.
            #default = field_obj.default if field_obj.default else field_obj.default_factory()
            vspec.kwargs['if_missing'] = PLACEHOLDER
            vspec.kwargs['if_empty'] = PLACEHOLDER
        elif vspec.factory == self.build or (not issubclass(vspec.factory, AnyMatch) and not issubclass(vspec.factory, PassThrough)):
            vspec.kwargs['not_empty'] = True
        if metadata:
            vspec.kwargs.update(metadata)
        if self.debug:
            logger.debug(f'Resolved {field_obj.name if field_obj is not NOT_SET else None}:{annotation}:{origin}:{a_args} to {v}(*{v_args}, **{v_kwargs})')
        return vspec.factory(*vspec.args if vspec.args else (), **(vspec.kwargs if vspec.kwargs else {}))

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


def pretty_print_schema(schema):
    pass


def main():
    msg = {
         "name":"alice",
         "groups": [
              "admin",
              "engineering"
         ],
         "email":None,
         "is_person": 7,
        'num_logins': 0,
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
    try:
        pprint(schema.to_python(msg))
    except Invalid as e:
        pprint (e.unpack_errors())


if __name__ == '__main__':
    main()
