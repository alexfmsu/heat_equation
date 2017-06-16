#ifndef _THOMAS_X_
#define _THOMAS_X_

extern int nx;

extern double** u_left;
extern double** u_right;

extern double dx2;
extern double dy2;

extern double* x;
extern double* y;

extern double dt;

extern double F(double x, double y, double t);

// BEGIN---------------------------------------- THOMAS ALGORITHM -------------------------------------------// ?
void solveMMatrix(const int n, const double *_A, const double *_B, const double *_C, double *f, double *x){
    // ---------------------------
    double A[n];
    double B[n];
    double C[n];
    
    for(int i = 0; i < n; i++){
        A[i] = _A[i];
        B[i] = _B[i];
        C[i] = _C[i];
    }
    // ---------------------------        
    double coeff;
    
    for(int i = 1; i < n; i++){
        coeff = A[i] / B[i-1];
        
        B[i] -= coeff * C[i-1];
        f[i] -= coeff * f[i-1];
    }

    x[n-1] = f[n-1] / B[n-1];
    
    for(int i = n-2; i >= 0; i--){
        x[i] = (f[i] - C[i] * x[i+1]) / B[i];
    }
}
// END------------------------------------------ THOMAS ALGORITHM -------------------------------------------// ?

extern double** u_half;

double** thomas_x(double** u, double* A_j, double* B_j, double* C_j, double* F_j, double* X_j, int k, int i, double t_half){
    #pragma omp parallel for
    for(int j = 1; j < nx; j++){
        F_j[j-1] = (u[i+1][j] - 2 * u[i][j] + u[i-1][j]) / dy2 + F(x[j], y[i], t_half) + 2 * u[i][j] / dt;            
    }
    
    F_j[0] += u_left[2*k+1][i] / dx2;
    F_j[nx-2] += u_right[2*k+1][i] / dx2;
    
    solveMMatrix(nx-1, A_j, B_j, C_j, F_j, X_j);
    
    for(int j = 0; j < nx-1; j++){
        u_half[i][j+1] = X_j[j];
    }
    
    return u_half;
}

#endif
