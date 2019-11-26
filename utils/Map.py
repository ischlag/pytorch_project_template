class Map(dict):
  """
  A map with where keys can be accessed using the "."

  Example:
  m = Map()
  m["a"] = 1
  m.b = 2
  print(m.a)  # 1
  print(m["b"])  # 2
  print(m.c)  # None
  print(m["c"])  # None
  """

  def __init__(self, *args, **kwargs):
      super(Map, self).__init__(*args, **kwargs)
      for arg in args:
          if isinstance(arg, dict):
              for k, v in arg.items():
                  self[k] = v

      if kwargs:
          for k, v in kwargs.items():
              self[k] = v

  def __getattr__(self, key):
    # is called when: m.a
    return self.get(key)

  def __setattr__(self, key, value):
    # is called when: m.a = 1
    self.__setitem__(key, value)

  def __getitem__(self, key):
    # is called when: m["b"]
    return self.get(key)

  def __setitem__(self, key, value):
    # is called when: m["b"] = 2
    super(Map, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, key):
    # is called when: del m.a
    self.__delitem__(key)

  def __delitem__(self, key):
    # is called when: del m["a"]
    super(Map, self).__delitem__(key)
    del self.__dict__[key]
