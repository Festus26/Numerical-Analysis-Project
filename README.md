# Numerical Analysis Project

A comprehensive C program that implements various numerical methods for solving mathematical problems.

## Features

### Root-Finding Methods
- **Bisection Method** - Finds roots by repeatedly bisecting an interval
- **Regula-Falsi Method** - Uses linear interpolation for root finding
- **Newton-Raphson Method** - Fast convergence using derivatives

### Linear System Solvers
- **Inverse Matrix** - Calculates matrix inverse using determinants and adjoints
- **Gaussian Elimination** - Solves systems of linear equations
- **Gauss-Seidel Method** - Iterative solver for linear systems

### Numerical Differentiation
- **Forward Difference**
- **Backward Difference**
- **Central Difference**

### Numerical Integration
- **Trapezoidal Rule**
- **Simpson's 1/3 Rule**
- **Simpson's 3/8 Rule**

### Interpolation
- **Gregory-Newton Interpolation** - Estimates values between known data points

## Expression Parser

The program includes a custom mathematical expression parser that supports:
- Basic operators: `+`, `-`, `*`, `/`, `^` (power)
- Trigonometric functions: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`
- Inverse trig functions: `asin`, `acos`, `atan`, `acot`, `asec`, `acsc`
- Exponential function: `exp`
- Logarithm function: `log`
- Variables and constants

## Compilation

```bash
gcc NumericalSolver.c -o numerical_solver -lm
```

## Usage

Run the compiled program:

```bash
./numerical_solver
```

Follow the interactive menu to select the desired numerical method.

## Expression Syntax

When entering mathematical expressions:
- Use **spaces** between tokens
- Use **parentheses** for every term to ensure correct operation order
- The minus operator `-` should be used carefully (see examples below)

### Examples

1. `( x ^ 2 ) + ( - 2 * x ) + 1`
2. `log 2 ( sin ( exp ( x ) + x ^ 2 ) )`
3. `1 / ( atan ( ( x ^ 2 ) - exp ( 1 ) ) )`

## Requirements

- C compiler (GCC recommended)
- Math library (`-lm` flag)

## License

This project is open source and available for educational purposes.
