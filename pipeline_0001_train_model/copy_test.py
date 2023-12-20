def func_01(a):
    a += 1
    return a


if __name__=='__main__':
    a = 0
    print('original:', a)

    func_01(a)
    print('1st func:', a)

    a = func_01(a)
    print('2nd func:', a)



def func_01(a):
    a[2] += 1

    return []


if __name__=='__main__':
    a = [1,2,3]
    print('original:', a)

    func_01(a)
    print('1st func:', a)
