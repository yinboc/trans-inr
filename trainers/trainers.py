trainers_dict = {}


def register(name):
    def decorator(cls):
        trainers_dict[name] = cls
        return cls
    return decorator
