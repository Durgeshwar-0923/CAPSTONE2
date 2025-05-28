from functools import reduce

# Define mathematical operations
add = lambda *args: sum(args)
mul = lambda *args: reduce(lambda x, y: x * y, args)
sub = lambda *args: reduce(lambda x, y: x - y, args)
div = lambda *args: reduce(lambda x, y: x // y, args)  
rem = lambda *args: reduce(lambda x, y: x % y, args)

new=lambda x,y,op:op(x,y)

operations = {
    '+': add,
    '*': mul,
    '-': sub,
    '/': div,
    '%': rem
}

while True:
    n = list(input("Enter input \n").split())
    
    if not n:
        print("Exit")
        break
    
    try:
        res = int(n[0])
        i = 1
        j = 2
        while i < len(n) - 1:
            num1 = int(n[j])
            op = n[i]
            res = new(res, num1, operations[op])
            i += 2
            j += 2
        print(res)
    
    except ValueError:
        print("Error: Please enter valid numbers.")
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
