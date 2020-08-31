def main():
    foo(1,2,3,a=1)

def foo(d, *arg, **arg2):
    print(arg)
    print(arg2)
    
if __name__ == '__main__':
    main()