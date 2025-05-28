import os
import shlex

def change_dir(filename):
    try:
        os.chdir(filename)
    except FileNotFoundError:
        print("Directory not found")

def ls():
    for dir in os.listdir():
        if os.path.isdir(dir):
            print(f"{dir}/", end=" ")  
        else:
            print(dir, end=" ")
    print()

def create(filename, lst=""):
    with open(filename, 'w') as file:
        file.write(lst)

def update(filename, lst):
    try:
        with open(filename, 'a+') as file:
            file.writelines('\n'.join(lst) + "\n")
            file.seek(0)
            data = file.read()
            print(data)
    except FileNotFoundError:
        print("File not found")

def remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("File is not found")

while True:
    user_input = shlex.split(input("Enter the input \n"))
    k = user_input.copy()

    def user_action(user_input):
        try:
            if not k:
                print("No command provided.")
                return

            action = k[0]
            filename = k[1] if len(k) >= 2 else None
            data = k[2] if len(k) >= 3 else ""

            if action == 'ls':
                ls()
            elif action == 'touch' and filename:
                create(filename, data)
            elif action == 'cd' and filename:
                change_dir(filename)
            else:
                print("Invalid command")
        except Exception as e:
            print(f"Error: {e}")

    user_action(user_input)
