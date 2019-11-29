class __Base:
    def __init__(self):
        print('children class')

    def math_method(self, n, a, b):
        return sum(a+b)/n

class Main(__Base):
    def __init__(self):
        print('parent class')

    @classmethod
    def act(cls, n=10, a=100, b=20, *args): # classmethod内ではselfを使えない
        return cls()

if __name__ == '__main__':
    nanika = Main.act()
