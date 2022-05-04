import sys 

print('Is this a real person? yes/no')
while True:
    input = sys.stdin.readline()
    if input.strip() == 'yes':
        print('input is', input)
        break
    elif input.strip() == 'no':   
        print('input is', input)
        break
    else: 
        print('Input wrong')