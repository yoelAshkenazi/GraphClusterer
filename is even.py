def generate_is_even_function(upper_limit):
    """
    Generates a Python file with an IsEven function that uses if statements
    to determine if a number is even or odd up to the specified upper limit.

    :param upper_limit: The maximum integer to handle in the IsEven function.
    """
    filename = 'even.py'
    with open(filename, 'w') as file:
        # Write the function definition
        file.write("def IsEven(number):\n")
        
        # Add the if statements for each number up to upper_limit
        for num in range(upper_limit + 1):
            parity = "Even" if num % 2 == 0 else "Odd"
            file.write(f"    if number == {num}:\n")
            file.write(f"        print(\"{parity}\")\n")
    
        print(f"File '{filename}' has been created with the IsEven function up to {upper_limit}.")

# Example usage with a smaller upper_limit to avoid performance issues:
generate_is_even_function(100)
