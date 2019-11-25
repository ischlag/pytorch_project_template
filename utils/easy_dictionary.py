class EasyDictionary:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def keys(self):
    return self.__dict__.keys()

  def __repr__(self):
    keys = self.__dict__.keys()
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}:\n\t{}".format(type(self).__name__, "\n\t".join(items))

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __getitem__(self, arg):
    return self.__dict__[arg]
