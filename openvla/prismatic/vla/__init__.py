def get_vla_dataset_and_collator(*args, **kwargs):
    """Load the optional RLDS dataset stack only when it is requested."""
    from .materialize import get_vla_dataset_and_collator as _materialize

    return _materialize(*args, **kwargs)
