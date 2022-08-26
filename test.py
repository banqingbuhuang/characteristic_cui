def hanoi(n, a, b, c):
    if n > 0:
        hanoi(n - 1, a, c, b)
        print(a+'->'+b)
        hanoi(n - 1, c, b, a)
def compare():
    Q = int(input())
    q = []
    for i in range(Q):
        q.append(list(input().split()))  # rstrip()删除字符串尾部的空字符
    print(q)

if __name__ == '__main__':
    # n = int(input())
    # hanoi(n, "a", "c", "b")
    # compare()
    m,n = map(int, input().split())  # 以空格间隔
    a = [int(x) for x in input().split()]
    b=[int(x) for x in input().split()]
    print(m,n)
    print(a)
    print(b)