#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define MAX_TKN_SIZE 100
#define MAX_INPUT_SIZE 100

// Enum to define the type of the node
typedef enum {
    CONSTANT,
    VARIABLE,
    OPERATOR,
    FUNCTION,
    LOGARITHM
} NodeType;

// Enum to define the type of the function
typedef enum {
    SIN, COS, TAN, COT, SEC, CSC, ASIN, ACOS, ATAN, ACOT, ASEC, ACSC, EXP
} FunctionType;

// Structure for a node
typedef struct Node {
    NodeType type;
    union {
        double value;  // for constants
        char variable; // for variable
        struct {
            char operator;
            struct Node* left;
            struct Node* right;
        } op;  // for operators
        struct {
            FunctionType function;
            struct Node* argument;
        } func;  // for functions
        struct {
            struct Node* base;
            struct Node* argument;
        } log;  // for logarithms
    } data;
} Node;

// Token structure
typedef struct {
    char** tokens;
    int size;
} TokenList;

// Parser function prototypes
TokenList tokenize(char* input);
Node* parseExpression(TokenList* tokens, int* pos);
Node* parsePrimary(TokenList* tokens, int* pos);
Node* parseTerm(TokenList* tokens, int* pos);
Node* parse(TokenList* tokens);
Node* parseLog(TokenList* tokens, int* pos);
Node* parseAST(TokenList tokens);
void freeNode(Node* node);
void printNode(Node* node);
double evaluate(Node* node, double x);

void checkAllocation(void* ptr);
void printMatrix(double **, int, int);

void chooseMethod(int);
void menu();

// Numeric method prototypes
void bisection();
void regulaFalsi();
void newtonRaphson();
void inverseMatix();
void gaussianElimination();
void gaussSeidel();
void numericDifferentiation();
void trapezoidalRule();
void simpsonRules();
void gregoryNewtonEnterpolation();

double forwardDifference(double, double, Node*);
double backwardDifference(double, double, Node*);
double centralDifference(double, double, Node*);

double simpson1_3Method(double, double, double, Node*);
double simpson3_8Method(double, double, double, Node*); // ( ( ( - 2 * x ) + 5 ) / 10 ) + ( - 3 * x ) )

int main() {
    while(1){
        menu();
    }
    return 0;
}
void menu(){
    int method = 0;
    printf("\n\tWhich method do you want to use?\n\t{-1: exit}\t{0: explanation}\n"); //4 2 3 1 9 4 5 7 10 2 3 7
    printf("1. Bisection Method\n");
    printf("2. Regula-Falsi Method\n");
    printf("3. Newton-Raphson Method\n");
    printf("4. Inverse Matrix\n");
    printf("5. Gaussian Elimination\n");
    printf("6. Gauss-Seidel\n");
    printf("7. Numeric Differentiation\n");
    printf("8. Simpson's Rules\n");
    printf("9. Trapezoidal Rule\n");
    printf("10. Gregory-Newton Enterpolation\n");
    scanf("%d", &method);
    chooseMethod(method);
}
void explanation(){
    printf("Parser and evaluator works for most of the functions and expressions.\n");
    printf("Parser needs spaces between the tokens.\n");
    printf("Use parenthesis for every term to get the perfect operation result.\n");
    printf("Minus operator '-' is likely to parsed wrong in some cases so use it as in the examples.\n");
    printf("Check the expression output to see if the expression is parsed correctly.\n");
    printf("Some examples:\n");
    printf("1. ( x ^ 2 ) + ( - 2 * x ) + 1\n");
    printf("2. log 2 ( sin ( exp ( x ) + x ^ 2 ) )\n");
    printf("3. 1 / ( atan ( ( x ^ 2 ) - exp ( 1 ) ) )\n");
}
void chooseMethod(int method){
    switch (method){
        case -1:
            exit(0);
            break;
        case 0:
            explanation();
            break;
        case 1:
            bisection();
            break;
        case 2:
            regulaFalsi();
            break;
        case 3:
            newtonRaphson();
            break;
        case 4:
            inverseMatix();
            break;
        case 5:
            gaussianElimination();
            break;
        case 6:
            gaussSeidel();
            break;
        case 7:
            numericDifferentiation();
            break;
        case 8:
            simpsonRules();
            break;
        case 9:
            trapezoidalRule();
            break;
        case 10:
            gregoryNewtonEnterpolation();
            break;
        default:
            break;
    }
}
void bisection(){
    double x1, x2, f1, f2, error;
    char input[MAX_INPUT_SIZE];// ( x ^ 3 ) + ( - 7 * ( x ^ 2 ) ) + ( 14 * x ) - 6
    getchar();
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);

    
    printf("Enter the initial values of x1 and x2:");
    scanf("%lf %lf", &x1, &x2);
    printf("Enter the error:");
    scanf("%lf", &error);
    error = fabs(error);
    f1 = evaluate(ast, x1);
    f2 = evaluate(ast, x2);
    printf("f1:%f\nf2:%f\n", f1, f2);

    printf("Enter the max number of iterations:\n");
    int maxIt;
    scanf("%d", &maxIt);

    double x0 = (x1 + x2) / 2;
    double f0 = evaluate(ast, x0);
    int i = 1;
    while (fabs((x2-x1)/pow(2,i)) > error && i < maxIt + 1){
        if (f1 * f0 < 0){
            x2 = x0;
            f2 = f0;
        }
        else{
            x1 = x0;
            f1 = f0;
        }
        printf("x1:%f\nx2:%f\n", x1, x2);
        printf("f1:%f\nf2:%f\n", f1, f2);
        x0 = (x1 + x2) / 2;
        f0 = evaluate(ast, x0);
        i++;

    }
    if(i == maxIt)
        printf("Max number of iterations done. This function may not converge at this value. \n");
    printf("The root of the function is: %f\n", x0);
    printf("The number of iterations: %d\n", i);
    // Free allocated memory
    freeNode(ast);
    for (i = 0; i < tokens.size; i++) {
        free(tokens.tokens[i]);
    }
    free(tokens.tokens);
}

void regulaFalsi(){
    double x1, x2, f1, f2, error;
    char input[MAX_INPUT_SIZE];// ( x ^ 3 ) + ( - 2  * ( x ^ 2 ) ) - 5
    getchar();
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);

    printf("Enter the initial values of x1 and x2:");
    scanf("%lf %lf", &x1, &x2);
    printf("Enter the error:");
    scanf("%lf", &error);
    error = fabs(error);
    f1 = evaluate(ast, x1);
    f2 = evaluate(ast, x2);
    printf("f1:%f\nf2:%f\n", f1, f2);
    printf("Enter the max number of iterations:\n");
    int maxIt;
    scanf("%d", &maxIt);

    double x0 = (x1 * f2 - x2 * f1) / (f2 - f1);
    double f0 = evaluate(ast, x0);
    int i = 1;
    while (fabs((x2-x1)/pow(2,i)) > error && i < maxIt + 1){
        if (f1 * f0 < 0){
            x2 = x0;
            f2 = f0;
        }
        else{
            x1 = x0;
            f1 = f0;
        }
        x0 = (x1 * f2 - x2 * f1) / (f2 - f1);
        f0 = evaluate(ast, x0);
        printf("x1:%f\nx2:%f\n", x1, x2);
        printf("f1:%f\nf2:%f\n", f1, f2);
        i++;
    }
    if(i == maxIt)
        printf("Max number of iterations done. This function may not converge at this value. \n");
    printf("The root of the function is: %f\n", x0);
    printf("The number of iterations: %d\n", i);
    // Free allocated memory
    freeNode(ast);
    for (i = 0; i < tokens.size; i++) {
        free(tokens.tokens[i]);
    }
    free(tokens.tokens);
}

void newtonRaphson() {  
    double x0, x1, f0, error, h = 0.000001;
    int iterations = 0, maxIt;
    char input[MAX_INPUT_SIZE];// ( x ^ 2 ) + ( - ( 4 * x ) ) + 3
    getchar();
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);
    
    printf("Enter the initial value of x0: ");
    scanf("%lf", &x0);
    printf("\n");
    f0 = evaluate(ast, x0);
    printf("f0:%f\n", f0);
    printf("Enter the max number of iterations:\n");
    scanf("%d", &maxIt);
    printf("Enter the error:");
    scanf("%lf", &error);
    error = fabs(error);
    printf("\n");
    double fd = centralDifference(x0, h, ast);
    double f = evaluate(ast, x0);
    x1 = x0 - (evaluate(ast, x0) / centralDifference(x0, h, ast));
    iterations++;
    printf("x%d:%f\nf(x%d):%lf\nf'(x%d):%lf\n", iterations, x1, iterations, f, iterations, fd);

    while(fabs(x1-x0) > error && iterations < maxIt + 1){
        x0 = x1;
        fd = centralDifference(x0, h, ast);
        f = evaluate(ast, x0);
        x1 = x0 - (f / fd);
        iterations++;
        printf("x%d:%f\nf(x%d):%lf\nf'(x%d):%lf\n", iterations, x1, iterations, f, iterations, fd);
    }
    if(iterations == maxIt)
        printf("Max number of iterations done. This function may not converge at this value. \n");
    printf("Root of the expression is: %lf\n", x1);
}

double **minor(double **matrix, int scale, int row, int column){
    int i, j;
    double **minorMatrix;
    minorMatrix = (double**)calloc(scale-1, sizeof(double*));
    checkAllocation(minorMatrix);
    for (i = 0; i < scale-1; i++){
        minorMatrix[i] = (double*)calloc(scale-1, sizeof(double));
        checkAllocation(minorMatrix[i]);
    }
    for (i = 0; i < scale; i++){
        for (j = 0; j < scale; j++){
            if (i < row && j < column){
                minorMatrix[i][j] = matrix[i][j];
            }
            else if (i < row && j > column){
                minorMatrix[i][j-1] = matrix[i][j];
            }
            else if (i > row && j < column){
                minorMatrix[i-1][j] = matrix[i][j];
            }
            else if (i > row && j > column){
                minorMatrix[i-1][j-1] = matrix[i][j];
            }
        }
    }
    return minorMatrix;
}
double determinant(double **matrix, int scale){
    int i;
    double det = 0;
    if (scale == 1){
        return matrix[0][0];
    }
    else if (scale == 2){
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
    }
    else{
        for (i = 0; i < scale; i++){
            det += pow(-1, i) * matrix[0][i] * determinant(minor(matrix, scale, 0, i), scale-1);
        }
        return det;
    }
}
double **adjoint(double **matrix, int scale){
    int i, j;
    double **adjointMatrix;
    adjointMatrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(adjointMatrix);
    for (i = 0; i < scale; i++){
        adjointMatrix[i] = (double*)calloc(scale, sizeof(double));
        checkAllocation(adjointMatrix[i]);
    }
    for (i = 0; i < scale; i++){
        for (j = 0; j < scale; j++){
            adjointMatrix[i][j] = pow(-1, i+j) * determinant(minor(matrix, scale, i, j), scale-1);
        }
    }
    return adjointMatrix;
}
void inverseMatix(){
    int i, j;
    int scale = 0;
    printf("Enter the scale(N) of the matrix:");
    scanf("%d", &scale);
    double **matrix;
    matrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(matrix);
    for (i = 0; i < scale; i++){
        matrix[i] = (double*)calloc(scale, sizeof(double));
        checkAllocation(matrix[i]);
    }
    printf("Enter the values of the matrix:\n");
    for (i = 0; i < scale; i++){
        for (j = 0; j < scale; j++){
            printf("[%d][%d]: ", i+1, j+1);
            scanf("%lf", &matrix[i][j]);
        }
    }
    printf("The matrix you entered is:\n");
    printMatrix(matrix, scale, scale);
    double **inverseMatrix;
    inverseMatrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(inverseMatrix);
    for (i = 0; i < scale; i++){
        inverseMatrix[i] = (double*)calloc(scale, sizeof(double));
        checkAllocation(inverseMatrix[i]);
    }
    double **adjointMatrix;
    adjointMatrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(adjointMatrix);
    for (i = 0; i < scale; i++){
        adjointMatrix[i] = (double*)calloc(scale, sizeof(double));
        checkAllocation(adjointMatrix[i]);
    }
    adjointMatrix = adjoint(matrix, scale);
    printf("The adjoint matrix is:\n");
    printMatrix(adjointMatrix, scale, scale);
    double det = determinant(matrix, scale);
    printf("The determinant of the matrix is: %f\n", det);
    if (det == 0){
        printf("The matrix is singular, it doesn't have an inverse.\n");
    }
    else{
        for (i = 0; i < scale; i++){
            for (j = 0; j < scale; j++){
                inverseMatrix[i][j] = adjointMatrix[j][i] / det;
            }
        }
        printf("The inverse matrix is:\n");
        printMatrix(inverseMatrix, scale, scale);
    }
}
void gaussianElimination(){
    int i, j, k;
    int scale = 0;
    printf("Enter the scale of the matrix:");
    scanf("%d", &scale);
    double **matrix;
    matrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(matrix);
    for (i = 0; i < scale; i++){
        matrix[i] = (double*)calloc(scale+1, sizeof(double));
        checkAllocation(matrix[i]);
    }
    checkAllocation(matrix);
    printf("\nEnter the values of the matrix:");
    for (i = 0; i < scale; i++){
        for (j = 0; j < scale+1; j++){
            printf(" [%d][%d]:", i+1, j+1);
            scanf("%lf", &matrix[i][j]);
        }
    }
    printf("\nThe matrix you entered is:\n");
    printMatrix(matrix, scale, scale + 1);
    if(determinant(matrix, scale) == 0){
        printf("The matrix is singular, it doesn't have a unique solution.\n");    
    }
    else{
        for (i = 0; i < scale -1 ; i++){
            for (j = i+1; j < scale; j++){
                if (i != j){
                    double ratio = matrix[j][i] / matrix[i][i];
                    for (k = 0; k < scale+1; k++){
                        matrix[j][k] -= ratio * matrix[i][k];
                    }
                }
            }
        }
        
        for (i = 0; i < scale; i++){
            double divisor = matrix[i][i];
            for (j = 0; j < scale+1; j++){
                matrix[i][j] /= divisor;
            }
        }
        
        
        double *solution = (double*)calloc(scale, sizeof(double));
        checkAllocation(solution);
        for (i = scale - 1; i >= 0; i--) {
            solution[i] = matrix[i][scale];
            for (j = i + 1; j < scale; j++) {
                solution[i] -= matrix[i][j] * solution[j];
            }
            solution[i] /= matrix[i][i];
        }
    
        printf("The matrix after Gaussian elimination is:\n");
        printMatrix(matrix, scale, scale + 1);
        printf("The solutions are:\n");
        for (i = 0; i < scale; i++){
            printf("%c%d = %f\n", 'x', i+1, solution[i]);
        }

        free(solution);
        for(i = 0; i < scale; i++)
            free(matrix[i]);
        free(matrix);
    }
}
// Function to make the matrix diagonally dominant 
int makeDiagonallyDominant(double **matrix, int scale) {
    int i, j, k, used[scale];
    for (i = 0; i < scale; i++)
        used[i] = 0;
    double **tempMatrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(tempMatrix);
    for (i = 0; i < scale; i++){
        tempMatrix[i] = (double*)calloc(scale+1, sizeof(double));
        checkAllocation(tempMatrix[i]);
    }
    for (i = 0; i < scale; i++) {
        int maxCol = 0;
        double maxVal = 0;
        for(j = 0; j < scale; j++){
            if(fabs(matrix[i][j]) >= fabs(maxVal)){
                maxCol = j;
                maxVal = matrix[i][j];
            }
        }
        if(used[maxCol] != 0)
            return 0;
        used[maxCol] = 1;
        for (k = 0; k < scale + 1; k++) {
            tempMatrix[maxCol][k] = matrix[i][k];
        }
    }
    for(i = 0; i < scale; i++){
        for(j = 0; j < scale + 1;j++)
            matrix[i][j] = tempMatrix[i][j];
        free(tempMatrix[i]);
    }
    free(tempMatrix);

    for(i = 0; i < scale; i++){
        if(used[i] == 0)
            return 0;
    }
    return 1;
}

void gaussSeidel(){
    int i, j;
    int scale = 0;
    printf("Enter the scale of the matrix:");
    scanf("%d", &scale);
    double **matrix;
    matrix = (double**)calloc(scale, sizeof(double*));
    checkAllocation(matrix);
    for (i = 0; i < scale; i++){
        matrix[i] = (double*)calloc(scale+1, sizeof(double));
        checkAllocation(matrix[i]);
    }
    checkAllocation(matrix);
    printf("The greatest values of each row should be in a different column!\n");
    printf("Enter the values of the matrix:\n");    //3.6 2.4 -1.8 6.3 4.2 -5.8 2.1 7.5 0.8 3.5 6.5 3.7 
    for (i = 0; i < scale; i++){
        for (j = 0; j < scale+1; j++){
            printf("[%d][%d]:", i+1, j+1);
            scanf("%lf", &matrix[i][j]);
        }
    }
    printf("The matrix you entered is:\n");
    printMatrix(matrix, scale, scale + 1); 

    if(determinant(matrix, scale) == 0){
        printf("The matrix is singular, it doesn't have a unique solution.\n");
        return;
    }
    if (!makeDiagonallyDominant(matrix, scale)) {
        printf("The matrix cannot be made diagonally dominant\n");
        return;
    }
    printf("The diagonally dominant matrix is:\n");
    printMatrix(matrix, scale, scale + 1);         
    double *solution;
    solution = (double*)calloc(scale, sizeof(double));
    double *newSolution;
    newSolution = (double*)calloc(scale, sizeof(double));
    checkAllocation(solution);
    for (i = 0; i < scale; i++){
        printf("Starting value of x%d:\n", i+1);
        scanf("%lf", &solution[i]);
        newSolution[i] = solution[i];
    }
    checkAllocation(newSolution);
    double error;
    printf("Enter the error:");
    scanf("%lf", &error);
    error = fabs(error);

    double maxError = error + 1;
    int iterations = 1;
    while (maxError > error){
        maxError = 0;
        for (i = 0; i < scale; i++){
            newSolution[i] = matrix[i][scale];
            for (j = 0; j < scale; j++){
                if (i != j){
                    newSolution[i] -= matrix[i][j] * solution[j];
                }
            }
            newSolution[i] /= matrix[i][i];
            if (fabs(newSolution[i] - solution[i]) > maxError){
                maxError = fabs(newSolution[i] - solution[i]);
            }
            printf("Iteration %d: [", iterations);
            for (j = 0; j < scale; j++){
                solution[j] = newSolution[j];
                printf("%lf", solution[j]);
                if(j != scale - 1)
                    printf(", ");
            }
            printf("]\n");
            iterations++;
        }
    }
    printf("The solutions are:\n");
    for (i = 0; i < scale; i++){
        printf("%c%d = %f\n", 'x', i+1, solution[i]);
    }
    for(i = 0; i < scale; i++)
        free(matrix[i]);
    free(matrix);

}

double forwardDifference(double x1, double h, Node* ast){
    double f1 = evaluate(ast, x1);
    double f2 = evaluate(ast, x1 + h);
    double derivative = (f2 - f1) / h;
    
    return derivative;
}
double backwardDifference(double x1, double h, Node* ast){
    double f1 = evaluate(ast, x1);
    double f2 = evaluate(ast, x1 - h);
    double derivative = (f1 - f2) / h;
    
    return derivative;
}
double centralDifference(double x1, double h, Node* ast){
    double f1 = evaluate(ast, x1 - h);
    double f2 = evaluate(ast, x1 + h);
    double derivative = (f2 - f1) / (2 * h);
    
    return derivative;
}

void numericDifferentiation(){
    int method = 0;
    do{
        printf("Which method do you want to use for differentiation?\n");
        printf("1. Forward Difference\n");
        printf("2. Backward Difference\n");
        printf("3. Central Difference\n");
        scanf("%d", &method);
        if(method < 1 || method > 3){
            printf("Invalid method\n");
        }
    }
    while(method < 1 || method > 3);

    double x1, h, derivative = 0;
    char input[MAX_INPUT_SIZE]; // ( x ^ 3 ) + ( - 7  * ( x ^ 2 ) ) + ( 14 * x ) - 6
    getchar();                
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);

    printf("Enter the value of x:");
    scanf("%lf", &x1);
    printf("\n");
    printf("Enter the value of h:");
    scanf("%lf", &h);
    printf("\n");

    switch (method){
        case 1:
            derivative = forwardDifference(x1, h, ast);
            printf("The forward derivative is: %lf\n", derivative);
            break;
        case 2:
            derivative = backwardDifference(x1, h, ast);
            printf("The backward derivative is: %lf\n", derivative);
            break;
        case 3:
            derivative = centralDifference(x1, h, ast);
            printf("The central derivative is: %lf\n", derivative);
            break;
        default:

            printf("Invalid method\n");
            break;
    }

    // Free allocated memory
    freeNode(ast);
    int i;
    for (i = 0; i < tokens.size; i++) {
        free(tokens.tokens[i]);
    }
    free(tokens.tokens);
}
double simpson1_3Method(double a, double b, double n, Node* ast){
    double h, sum = 0;
    int i;
    h = (b - a) / n;
    for (i = 1; i < n; i++){
        if (i % 2 == 0){
            sum += 2 * evaluate(ast, a + i * h);
        }
        else{
            sum += 4 * evaluate(ast, a + i * h);
        }
    }
    sum = h / 3 * ( evaluate(ast, a) + evaluate(ast, b) + sum);
    return sum;
}
double simpson3_8Method(double a, double b, double n, Node* ast){
    double h0, sum = 0;
    int i;
    h0 = (b - a) / n;
    for(i = 0; i < n; i++){
        b = a + h0;
        double h = (b - a) / 3;
        sum += (h * 3)/8 * ( evaluate(ast, a) + 3 * evaluate(ast, a + h) + 3 * evaluate(ast, a + 2 * h) + evaluate(ast, a + 3 * h));
        a = b;
    }
    return sum;
}
void simpsonRules(){
    int method = 0;
    double sum = 0, a, b, n;
    do{
        printf("Which method do you want to use for differentiation?\n");
        printf("1. Simpson 1/3 Rule\n");
        printf("2. Simpson 3/8 Rule\n");
        scanf("%d", &method);
        if(method < 1 || method > 2){
            printf("Invalid method\n");
        }
    }
    while(method < 1 || method > 2);
    char input[MAX_INPUT_SIZE];// ( ( x ^ 2 ) - 1 ) * ( x + 2 )
    getchar();                 
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);

    printf("Enter the values of a and b:");
    scanf("%lf %lf", &a, &b);
    printf("Enter the number of intervals:");
    scanf("%lf", &n);

    switch (method){
        case 1:
            sum = simpson1_3Method(a, b, n, ast);
            printf("The 1/3 Simpson integral is: %f\n", sum);
            break;
        case 2:
            sum = simpson3_8Method(a, b, n, ast);
            printf("The 3/8 Simpson integral is: %f\n", sum);
            break;
        default:
            printf("Invalid method\n");
            break;
    }
    // Free allocated memory
    freeNode(ast);
    int i;
    for (i = 0; i < tokens.size; i++) {
        free(tokens.tokens[i]);
    }
    free(tokens.tokens);
}
void trapezoidalRule(){
    double a, b, h, n, sum = 0;
    int i;
    char input[MAX_INPUT_SIZE];// 1 / ( 1 + ( x ^ 2 ) )
    getchar();
    printf("Enter the expression:");
    fgets(input, MAX_INPUT_SIZE, stdin);
    TokenList tokens = tokenize(input); // Tokenize the input

    Node* ast = parseAST(tokens);

    printf("Enter the values of a and b:");
    scanf("%lf %lf", &a, &b);
    printf("Enter the number of intervals:");
    scanf("%lf", &n);
    h = (b - a) / n;
    for (i = 1; i < n; i++){
        sum += evaluate(ast, a + i * h);
    }
    sum = h  * (( evaluate(ast, a) + evaluate(ast, b) ) / 2 + sum);
    printf("The integral is: %f\n", sum);
    // Free allocated memory
    freeNode(ast);
    for (i = 0; i < tokens.size; i++) {
        free(tokens.tokens[i]);
    }
    free(tokens.tokens);
}

void gregoryNewtonEnterpolation(){
    int i, j, n, diffCount = 0;
    printf("Enter the number of data points:");
    scanf("%d", &n);
    double x[n], y[n], h[n][n], x1;
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            h[i][j] = 0;
        }
    }
    printf("Enter the values of x and y:\n");
    for (i = 0; i < n; i++){
        printf("x%d:", i+1);
        scanf("%lf", &x[i]);
        printf("\n");
        printf("y%d:", i+1);
        scanf("%lf", &y[i]);
        printf("\n");
    }
    printf("Enter the value of x:");
    scanf("%lf", &x1);
    printf("\n");
    for (i = 0; i < n-1; i++){
        h[i][0] = y[i+1] - y[i];
    }
    diffCount = 1;
    for (i = 1; i < n-1; i++){
        int tempCount = 0;
        for (j = 0; j < n-i-1; j++){
            h[j][i] = h[j+1][i-1] - h[j][i-1]; // 0 -4 1 -2 2 14 3 62 4 160 5 326 6 578
            printf("%d\n", j);                 
            if(h[j][i] != 0)                   
                tempCount++;
        }
        if(tempCount != 0)
            diffCount++;
    }
    printf("The difference table is:\n");
    printf("x\t\ty\t\t");
    for (i = 0; i < diffCount; i++){
        printf("h%d\t\t", i+1);
    }
    printf("\n");
    for (i = 0; i < n; i++){
        printf("%lf\t%lf\t", x[i], y[i]);
        for (j = 0; j < diffCount; j++){
            if(h[i][j] != 0)
                printf("%lf\t", h[i][j]);
        }
        printf("\n");
    }
    double sum = y[0];
    printf("The expression is: %lf ", y[0]);
    for(i = 0; i < diffCount; i++){
        double term = 1;
        printf("+ (");
        for(j = 0; j < i + 1; j++){
            term *= (x1 - x[j]) / (j + 1);
            printf("((x - %d) / %d)", j, j + 1);
        }
        term *= 1 / (pow(x[1]-x[0], i+1));
        printf("(1/%lf), ", pow(x[1]-x[0], i+1));
        printf(") * %lf ", h[0][i]);
        term *= h[0][i];
        sum += term;
    }
    printf("\nThe value of y at x = %lf is: %lf\n", x1, sum);
}
void printMatrix(double **matrix, int scaleRow, int scaleColumn){
    int i, j;
    for (i = 0; i < scaleRow; i++){
        printf("[");
        for (j = 0; j < scaleColumn; j++){
            printf("%f", matrix[i][j]);
            if( j != scaleColumn - 1)
                printf(", ");
        }
        printf("]\n");
    }
}
Node* parseAST(TokenList tokens){
    if (tokens.size == 0) {
        fprintf(stderr, "Error: No tokens parsed\n");
        exit(1);
    }
    Node* ast = parse(&tokens);// Parse the tokens to create the AST
    if (!ast) {
        fprintf(stderr, "Error: Failed to parse the expression\n");
        int i;
        for (i = 0; i < tokens.size; i++) {
            free(tokens.tokens[i]);
        }
        free(tokens.tokens);
        exit(1);
    }

    printf("Expression: ");
    printNode(ast);
    printf("\n");
    return ast;
}
// Function to check memory allocation
void checkAllocation(void* ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
}

// Tokenizer to split the input into tokens
TokenList tokenize(char* input) {
    TokenList tokenList;
    tokenList.size = 0;
    tokenList.tokens = malloc(MAX_TKN_SIZE * sizeof(char*));
    checkAllocation(tokenList.tokens);
    char* token = strtok(input, " ");
    while (token != NULL) {
        tokenList.tokens[tokenList.size++] = strdup(token);
        token = strtok(NULL, " ");
    }

    return tokenList;
}

// Parsing logarithm function
Node* parseLog(TokenList* tokens, int* pos) {
    //printf("Parsing logarithm\n");
    (*pos)++; // Skip "log" token
    if (*pos >= tokens->size) {
        fprintf(stderr, "Error: Unexpected end of input after 'log'\n");
        return NULL;
    }
    Node* base = parsePrimary(tokens, pos);
    if (base == NULL) {
        fprintf(stderr, "Error: Invalid base for logarithm\n");
        return NULL;
    }
    (*pos)++; // Skip base token
    if (*pos >= tokens->size) {
        fprintf(stderr, "Error: Unexpected base for logarithm'\n");
        freeNode(base);
        return NULL;
    }
    Node* argument = parseExpression(tokens, pos);
    if (argument == NULL) {
        fprintf(stderr, "Error: Invalid argument for logarithm\n");
        freeNode(base);
        return NULL;
    }
    // Create a new node for logarithm
    Node* node = malloc(sizeof(Node));
    checkAllocation(node);
    node->type = LOGARITHM;
    node->data.log.base = base;
    node->data.log.argument = argument;
    return node;
}
// Parsing primary expression (constants, variables, functions, parentheses)
Node* parsePrimary(TokenList* tokens, int* pos) {
    if (*pos >= tokens->size) {
        return NULL;
    }

    char* token = tokens->tokens[*pos];
    // Check if the token is a function
    if (strncmp(token, "sin", 3) == 0 || strncmp(token, "cos", 3) == 0 || 
        strncmp(token, "tan", 3) == 0 || strncmp(token, "asin", 4) == 0 || 
        strncmp(token, "acos", 4) == 0 || strncmp(token, "atan", 4) == 0 || 
        strncmp(token, "exp", 3) == 0 || strncmp(token, "cot", 3) == 0 || 
        strncmp(token, "sec", 3) == 0 || strncmp(token, "csc", 3) == 0 ||
        strncmp(token, "acot", 4) == 0 || strncmp(token, "asec", 4) == 0 ||
        strncmp(token, "acsc", 4) == 0){
        // Parse the function
        (*pos)++;
        Node* node = malloc(sizeof(Node));
        checkAllocation(node);
        node->type = FUNCTION;
        // Set the function type
        if (strncmp(token, "sin", 3) == 0) node->data.func.function = SIN;
        else if (strncmp(token, "cos", 3) == 0) node->data.func.function = COS;
        else if (strncmp(token, "tan", 3) == 0) node->data.func.function = TAN;
        else if (strncmp(token, "asin", 4) == 0) node->data.func.function = ASIN;
        else if (strncmp(token, "acos", 4) == 0) node->data.func.function = ACOS;
        else if (strncmp(token, "atan", 4) == 0) node->data.func.function = ATAN;
        else if (strncmp(token, "exp", 3) == 0) node->data.func.function = EXP;
        else if (strncmp(token, "cot", 3) == 0) node->data.func.function = COT;
        else if (strncmp(token, "sec", 3) == 0) node->data.func.function = SEC;
        else if (strncmp(token, "csc", 3) == 0) node->data.func.function = CSC;
        else if (strncmp(token, "acot", 4) == 0) node->data.func.function = ACOT;
        else if (strncmp(token, "asec", 4) == 0) node->data.func.function = ASEC;
        else if (strncmp(token, "acsc", 4) == 0) node->data.func.function = ACSC;
        // Parse the argument
        node->data.func.argument = parseExpression(tokens, pos);
        return node;
    } else if (strncmp(token, "log", 3) == 0){ // Check if the token is a logarithm
        return parseLog(tokens, pos);
    } else if (isdigit(token[0]) || token[0] == '.') { // Check if the token is a number
        (*pos)++;
        Node* node = malloc(sizeof(Node));
        checkAllocation(node);
        node->type = CONSTANT;
        node->data.value = atof(token);
        return node;
    } else if (isalpha(token[0])) { // Check if the token is a variable
        (*pos)++;
        Node* node = malloc(sizeof(Node));
        checkAllocation(node);
        node->type = VARIABLE;
        node->data.variable = token[0];
        return node;
    } else if (token[0] == '(') { // Check if the token is an expression in parentheses
        (*pos)++;
        Node* node = parseExpression(tokens, pos);
        if (*pos < tokens->size && tokens->tokens[*pos][0] == ')') {
            (*pos)++;
            return node;
        } else { 
            fprintf(stderr, "Error: Missing closing parenthesis\n");
            freeNode(node);
            return NULL;
        }
    }

    //fprintf(stderr, "Error: Invalid token: %s\n", token);
    return NULL;
}
// Parsing term expression (multiplication, division, exponentiation)
Node* parseTerm(TokenList* tokens, int* pos) {
    Node* node = parsePrimary(tokens, pos);

    while (*pos < tokens->size) {
        char* token = tokens->tokens[*pos];
        if (strcmp(token, "^") == 0) {
            (*pos)++;
            Node* right = parseExpression(tokens, pos);
            Node* newNode = malloc(sizeof(Node));
            checkAllocation(newNode);
            newNode->type = OPERATOR;
            newNode->data.op.operator = '^';
            newNode->data.op.left = node;
            newNode->data.op.right = right;
            node = newNode;
        } else if (strcmp(token, "*") == 0 || strcmp(token, "/") == 0) {
            (*pos)++;
            Node* right = parseExpression(tokens, pos);
            Node* newNode = malloc(sizeof(Node));
            checkAllocation(newNode);
            newNode->type = OPERATOR;
            newNode->data.op.operator = token[0];
            newNode->data.op.left = node;
            newNode->data.op.right = right;
            node = newNode;
        } else {
            break;
        }
    }

    return node;
}
// Parsing expression (addition, subtraction)
Node* parseExpression(TokenList* tokens, int* pos) {
    Node* node = parseTerm(tokens, pos);

    while (*pos < tokens->size) {
        char* token = tokens->tokens[*pos];
        if (strcmp(token, "+") == 0 || strcmp(token, "-") == 0) {
            (*pos)++;
            Node* right = parseExpression(tokens, pos);
            Node* newNode = malloc(sizeof(Node));
            checkAllocation(newNode);
            newNode->type = OPERATOR;
            newNode->data.op.operator = token[0];
            newNode->data.op.left = node;
            newNode->data.op.right = right;
            node = newNode;
        } else {
            break;
        }
    }

    return node;
}
// Parsing function to create the AST
Node* parse(TokenList* tokens) {
    int pos = 0;
    return parseExpression(tokens, &pos);
}

// Evaluation function
double evaluate(Node* node, double x) {
    if (!node) {
        //fprintf(stderr, "Error: Null node encountered during evaluation\n");
        return 0;
    }
    // Evaluate the node based on its type
    switch (node->type) {
        case CONSTANT:
            return node->data.value;
        case VARIABLE:
            return x;
        case OPERATOR: {
            double left = evaluate(node->data.op.left, x);
            double right = evaluate(node->data.op.right, x);
            switch (node->data.op.operator) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return left / right;
                case '^': return pow(left, right);
                default:
                    fprintf(stderr, "Error: Unknown operator %c\n", node->data.op.operator);
                    return 0;
            }
        }
        case FUNCTION: {
            double arg = evaluate(node->data.func.argument, x);
            switch (node->data.func.function) {
                case SIN: return sin(arg);
                case COS: return cos(arg);
                case TAN: return tan(arg);
                case ASIN: return asin(arg);
                case ACOS: return acos(arg);
                case ATAN: return atan(arg);
                case EXP: return exp(arg);
                case COT: return 1 / tan(arg);
                case SEC: return 1 / cos(arg);
                case CSC: return 1 / sin(arg);
                case ACOT: return atan(1 / arg);
                case ASEC: return acos(1 / arg);
                case ACSC: return asin(1 / arg);
                default:
                    fprintf(stderr, "Error: Unknown function\n");
                    return 0;
            }
        }
        case LOGARITHM: {
            double base = evaluate(node->data.log.base, x);
            double argument = evaluate(node->data.log.argument, x);
            if (base <= 0 || base == 1 || argument <= 0) {
                fprintf(stderr, "Error: Invalid base or argument for logarithm\n");
                return 0;
            }
            return log(argument) / log(base);
        }
        default:
            fprintf(stderr, "Error: Unknown node type\n");
            return 0;
    }
    return 0;
}

// Free the AST
void freeNode(Node* node) {
    if (node == NULL) return;
    switch (node->type) {
        case OPERATOR:
            freeNode(node->data.op.left);
            freeNode(node->data.op.right);
            break;
        case FUNCTION:
            freeNode(node->data.func.argument);
            break;
        case LOGARITHM:
            freeNode(node->data.log.base);
            freeNode(node->data.log.argument);
            break;
        default:
            break;
    }
    free(node);
}

// Debugging function to print the AST
void printNode(Node* node) {
    if (!node) return;

    switch (node->type) {
        case CONSTANT:
            printf("%f", node->data.value);
            break;
        case VARIABLE:
            printf("%c", node->data.variable);
            break;
        case OPERATOR:
            printf("(");
            printNode(node->data.op.left);
            printf(" %c ", node->data.op.operator);
            printNode(node->data.op.right);
            printf(")");
            break;
        case FUNCTION:
            switch (node->data.func.function) {
                case SIN: printf("sin("); break;
                case COS: printf("cos("); break;
                case TAN: printf("tan("); break;
                case ASIN: printf("asin("); break;
                case ACOS: printf("acos("); break;
                case ATAN: printf("atan("); break;
                case EXP: printf("exp("); break;
                case COT: printf("cot("); break;
                case SEC: printf("sec("); break;
                case CSC: printf("csc("); break;
                case ACOT: printf("acot("); break;
                case ASEC: printf("asec("); break;
                case ACSC: printf("acsc("); break;
                default: printf("unknown("); break;
            }
            printNode(node->data.func.argument);
            printf(")");
            break;
        case LOGARITHM:
            printf("log_");
            printNode(node->data.log.base);
            printf("(");
            printNode(node->data.log.argument);
            printf(")");
            break;
        default:
            printf("unknown");
            break;
    }
}