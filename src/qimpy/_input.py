from typing import Optional, List, Union, TypeVar, Type, NamedTuple


ClassType = TypeVar('ClassType')
ConstructableType = TypeVar('ConstructableType', bound="Constructable")
ConstructableType2 = TypeVar('ConstructableType2', bound="Constructable")


class ConstructOptions(NamedTuple):
    """Options passed through `__init__` of all Constructable objects."""
    parent: Optional['Constructable'] = None  #: Parent in heirarchy
    attr_name: str = ''  #: Attribute name of object within parent


class Constructable:
    """Base class of dict-constructable and serializable objects
    in QimPy heirarchy."""
    __slots__ = ('parent', 'children', 'path')
    parent: Optional['Constructable']  #: Parent object in heirarchy (if any)
    children: List['Constructable']  #: Child objects in heirarchy
    path: List[str]  #: Elements of object's absolute path in heirarchy

    def __init__(self, co: ConstructOptions, **kwargs):
        self.parent = co.parent
        self.children = []
        self.path = ([] if (co.parent is None)
                     else (co.parent.path + [co.attr_name]))

    @classmethod
    def construct(cls: Type[ConstructableType],
                  parent: ConstructableType2, attr_name: str,
                  params: Union[ConstructableType, dict, None],
                  attr_version_name: str = '', **kwargs) -> None:
        """Construct object of type `cls` in QimPy heirarchy.
        Set the result as attribute named `attr_name` of `parent`.
        Specifically, construct object from `params` and `kwargs`
        if `params` is a dict, and just from `kwargs` if `params` is None.
        Any '-' in the keys of `params` are replaced with '_' for convenience.
        Otherwise check that `params` is already of type `cls`, and if not,
        raise an error clearly stating the types `attr_name` can be.

        Optionally, `attr_version_name` overrides `attr_name` used in the
        error, which may be useful when the same `attr_name` could be
        initialized by several versions eg. `kpoints` in :class:`Electrons`
        could be `k-mesh` (:class:`Kmesh`) or `k-path` (:class:`Kmesh`).
        """

        # Try all the valid possibilities:
        co = ConstructOptions(parent=parent, attr_name=attr_name)
        if isinstance(params, dict):
            result = cls(**kwargs, **dict_input_cleanup(params), co=co)
        elif params is None:
            result = cls(**kwargs, co=co)
        elif isinstance(params, cls):
            result = params
            Constructable.__init__(result, co=co)
        else:
            # Report error with canonicalized class name:
            module = cls.__module__
            module_elems = ([] if module is None else (
                [elem for elem in module.split('.')
                 if not elem.startswith('_')]))  # drop internal module names
            module_elems.append(cls.__qualname__)
            class_name = '.'.join(module_elems)
            a_name = (attr_version_name if attr_version_name else attr_name)
            raise TypeError(f'{a_name} must be dict or {class_name}')

        # Add to parent and set up links:
        setattr(parent, attr_name, result)
        parent.children.append(result)


def dict_input_cleanup(params: dict) -> dict:
    """Clean-up dict for use in constructors.
    This is required eg. for dicts from YAML to make sure keys are compatible
    with passing as keyword-only arguments to constructors. Most importantly,
    replace hyphens (which look nicer) in all keys to underscores internally,
    so that they become valid identifiers within the code"""
    return dict((k.replace('-', '_'), v) for k, v in params.items())
