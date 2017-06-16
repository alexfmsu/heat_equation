#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <mpi.h>

using namespace std;

// ------------------------
char* filename;
// ------------------------
double a1, b1;
double a2, b2;

double t_max;

int nx, ny, nt;

double dx, dy, dt;
double dx2, dy2;
double dt_1_2;

double* x;
double* y;
double* t;

double** u;
double** u_half;
// ------------------------
double *A_i, *B_i, *C_i;
double *A_j, *B_j, *C_j;
// ------------------------
double* F_i;
double* F_j;

double* X_i;
double* X_j;
// ------------------------
void print_grid();

void print_x();
void print_y();
void print_t();

void print_u();
void print_u_half();
void print_u_exact(int);
void print_error();

void free_memory();
// ------------------------

// BEGIN---------------------------------------- EXACT SOLUTION ---------------------------------------------
double u_exact(double x, double y, double t){
    // return exp(-t) * sin(M_PI * x) * cos(M_PI * y);
    return exp(-t) * (sin(M_PI * x) + cos(M_PI * y)) + 10;
}
// END------------------------------------------ EXACT SOLUTION ---------------------------------------------

// BEGIN---------------------------------------- F(x, y, t) -------------------------------------------------
double F(double x, double y, double t){
    // return exp(-t) * sin(M_PI * x) * cos(M_PI * y) * (2 * M_PI * M_PI - 1);
    return - exp(-t) * (sin(M_PI * x) + cos(M_PI * y)) * (1 - M_PI * M_PI);
}
// END------------------------------------------ F(x, y, t) -------------------------------------------------

// BEGIN---------------------------------------- READ DATA --------------------------------------------------
void read_data(){
    FILE* f;
    
    f = fopen(filename, "r");
    
    if(f == NULL){
        cout << "Cannot open file \"" << filename << "\"" << endl;
        
        exit(1);
    }
    
    int count = fscanf(f, "%lf %lf %lf %lf %lf %d %d %d", &a1, &b1, &a2, &b2, &t_max, &nx, &ny, &nt);
    
    if(count < 8){
        cout << "Wrong data format in file \"" << filename << "\"" << endl << endl;
        cout << "Usage: a1 b1 a2 b2" << endl;
        cout << "       t_max" << endl;
        cout << "       nx ny nt" << endl;
        cout << endl;
        
        cout << "Example: 0 1 0 1" << endl;
        cout << "         1" << endl;
        cout << "         10 10 5" << endl;
        cout << endl;
        
        exit(1);
    }
    
    fclose(f);
}
// END------------------------------------------ READ DATA --------------------------------------------------

// BEGIN---------------------------------------- THOMAS ALGORITHM -------------------------------------------
void solveMatrix(const int n, const double *_A, const double *_B, const double *_C, double *f, double *x){
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
// END------------------------------------------ THOMAS ALGORITHM -------------------------------------------

// BEGIN---------------------------------------- SET TRIAG COEFF --------------------------------------------
void set_triag_coeff(){
    A_i = new double[ny];
    B_i = new double[ny];
    C_i = new double[ny];
    
    for(int i = 0; i < ny; i++){
        A_i[i] = -1.0 / dy2;
        B_i[i] = 2.0 / dt + 2.0 / dy2;
        C_i[i] = -1.0 / dy2;
    }
    
    A_i[0] = 0;
    C_i[ny-1] = 0;
    // -------------------------------
    A_j = new double[nx];
    B_j = new double[nx];
    C_j = new double[nx];
    
    for(int i = 0; i < nx; i++){
        A_j[i] = -1.0 / dx2;
        B_j[i] = 2.0 / dt + 2.0 / dx2;
        C_j[i] = -1.0 / dx2;
    }
    
    A_j[0] = 0;
    C_j[nx-1] = 0;
}
// END------------------------------------------ SET TRIAG COEFF --------------------------------------------

// BEGIN---------------------------------------- SET GRID ---------------------------------------------------
void set_grid(){
    read_data();
    
    assert(a1 < b1);
    assert(a2 < b2);
    
    assert(t_max > 0);
    
    assert(nx > 0);
    assert(ny > 0);
    assert(nt > 0);
    
    dx = (b1 - a1) / nx;
    dy = (b2 - a2) / ny;
    dt = t_max / nt;
    
    dx2 = dx * dx;
    dy2 = dy * dy;
    
    dt_1_2 = dt / 2.0;
    
    try{
        x = new double[nx+1];
        y = new double[ny+1];
        t = new double[nt+1];
        
        u = new double*[ny+1];
        u_half = new double*[ny+1];
        
        for(int i = 0; i <= ny; i++){
            u[i] = new double[nx+1];
            u_half[i] = new double[nx+1];
            
            for(int j = 0; j <= nx; j++){
                u[i][j] = 0;
                u_half[i][j] = 0;
            }
        }
    }catch(std::bad_alloc){
        cout << "Cannot allocate memory" << endl;
        
        exit(1);
    }
    
    for(int i = 0; i <= ny; i++){
        y[i] = a2 + i * dy;
    }
    
    for(int j = 0; j <= nx; j++){
        x[j] = a1 + j * dx;
    }
    
    for(int k = 0; k <= nt; k++){
        t[k] = k * dt;
    }
}
// END------------------------------------------ SET GRID ---------------------------------------------------

// BEGIN---------------------------------------- INITIAL CONDITIONS -----------------------------------------
double g_0(int j, int i){
    return u_exact(x[j], y[i], t[0]);
}

void set_initials(){
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            u[i][j] = g_0(j, i);
        }
    }
}
// END------------------------------------------ INITIAL CONDITIONS -----------------------------------------

// BEGIN---------------------------------------- BOUNDARY CONDITIONS ----------------------------------------
void set_boundaries(double** _u, double _t){
    for(int i = 0; i <= ny; i++){
        _u[i][0] = u_exact(x[0], y[i], _t);
        _u[i][nx] = u_exact(x[nx], y[i], _t);
    }
    
    for(int j = 0; j <= nx; j++){
        _u[0][j] = u_exact(x[j], y[0], _t);
        _u[ny][j] = u_exact(x[j], y[ny], _t);
    }
}
// END------------------------------------------ BOUNDARY CONDITIONS ----------------------------------------

// BEGIN---------------------------------------- SOLVE ------------------------------------------------------
void solve(){
    set_triag_coeff();

    F_i = new double[ny-1];
    F_j = new double[nx-1];
    
    X_i = new double[ny-1];
    X_j = new double[nx-1];
    
    set_initials();
    
    clock_t t1 = clock();
    
    for(int k = 0; k < nt; k++){
        double t_half = t[k] + dt_1_2;
        double t_k = t_half + dt_1_2;
            
        // BEGIN-------------------------------- x-direction ------------------------------------------------
        set_boundaries(u_half, t_half);
        
        for(int i = 1; i < ny; i++){
            for(int j = 1; j < nx; j++){
                F_j[j-1] = (u[i+1][j] - 2 * u[i][j] + u[i-1][j]) / dy2 + F(x[j], y[i], t_half) + 2 * u[i][j] / dt;
            
                if(j == 1){
                    F_j[j-1] += u_exact(x[j-1], y[i], t_half) / dx2;
                }else if(j == nx-1){
                    F_j[j-1] += u_exact(x[j+1], y[i], t_half) / dx2;
                }
            }
            
            solveMatrix(nx-1, A_j, B_j, C_j, F_j, X_j);
            
            for(int j = 0; j < nx-1; j++){
                u_half[i][j+1] = X_j[j];
            }
        }
        // END---------------------------------- x-direction ------------------------------------------------
        
        // BEGIN-------------------------------- y-direction ------------------------------------------------
        set_boundaries(u, t_k);
        
        for(int j = 1; j < nx; j++){
            for(int i = 1; i < ny; i++){
                F_i[i-1] = (u_half[i][j-1] - 2 * u_half[i][j] + u_half[i][j+1]) / dx2 + F(x[j], y[i], t_half) + 2 * u_half[i][j] / dt;
            }
            
            F_i[0] += u_exact(x[j], y[0], t_k) / dy2;
            F_i[ny-2] += u_exact(x[j], y[ny], t_k) / dy2;
            
            solveMatrix(ny-1, A_i, B_i, C_i, F_i, X_i);
            
            for(int i = 0; i < ny-1; i++){
                u[i+1][j] = X_i[i];
            }
        }
        // END---------------------------------- y-direction ------------------------------------------------
    }
    
    clock_t t2 = clock();
    
    cout << "Time: " << (double)(t2 - t1) / (double)CLOCKS_PER_SEC << endl << endl;
}
// END------------------------------------------ SOLVE ------------------------------------------------------



int main(int argc, char** argv){
    cout << fixed << setprecision(8);
    
    if(argc < 2){
        cout << "Usage: ./a.out data" << endl;
        
        exit(1);
    }
    
    filename = argv[1];
    
    set_grid();
    
    solve();
    
    print_error();
    
    free_memory();
    
    return 0;
}



// BEGIN---------------------------------------- PRINT GRID -------------------------------------------------
void print_x(){
    cout << "x: ";
    
    for(int j = 0; j <= nx; j++){
        cout << x[j] << " ";
    }
    
    cout << endl << endl;
}

void print_y(){
    cout << "y: ";
    
    for(int i = 0; i <= ny; i++){
        cout << y[i] << " ";
    }
    
    cout << endl << endl;
}

void print_t(){
    cout << "t: ";
    
    for(int k = 0; k <= nt; k++){
        cout << t[k] << " ";
    }
    
    cout << endl << endl;
}

void print_grid(){
    print_x();
    print_y();
    print_t();
}
// END------------------------------------------ PRINT GRID -------------------------------------------------

// BEGIN---------------------------------------- PRINT SOLUTION ---------------------------------------------
void print_u(){
    cout << "u:" << endl;
    
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            cout << setw(12) << setprecision(8) << u[i][j] << " ";
        }
        
        cout << endl;
    }
    
    cout << endl;
}

void print_u_half(){
    cout << "u_half:" << endl;
    
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            cout << setw(12) << setprecision(9) << u_half[i][j] << " ";
        }
        
        cout << endl;
    }
    
    cout << endl;
}

void print_u_exact(int k){
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            cout << u_exact(x[j], y[i], t[k]) << " ";
        }
        
        cout << endl;
    }
    
    cout << endl;
}
// END------------------------------------------ PRINT SOLUTION ---------------------------------------------

// BEGIN---------------------------------------- PRINT ERROR ------------------------------------------------
void print_error(){
    double max_abs_error = 0;
    double max_rel_error = 0;
    
    for(int i = 1; i < ny; i++){
        for(int j = 1; j < nx; j++){
            double sln = u[i][j];
            
            double sln_e = u_exact(x[j], y[i], t[nt]);
            
            if(abs(sln_e) < 1e-10 || abs(sln) < 1e-10){
                continue;
            }

            double abs_error = abs(sln - sln_e);
            double rel_error = abs_error / abs(sln_e);
            
            max_abs_error = max(max_abs_error, abs_error);
            max_rel_error = max(max_rel_error, rel_error);
        }
    }
    
    cout << "Abs. error: " << setprecision(8) << max_abs_error << endl;
    cout << "Rel. error: " << setprecision(8) << max_rel_error << endl;
}
// END------------------------------------------ PRINT ERROR ------------------------------------------------

// BEGIN---------------------------------------- FREE MEMORY ------------------------------------------------
void free_memory(){
    // -------------------------
    delete[] x;
    delete[] y;
    delete[] t;
    // -------------------------
    delete[] A_i;
    delete[] B_i;
    delete[] C_i;
    // -------------------------
    delete[] A_j;
    delete[] B_j;
    delete[] C_j;
    // -------------------------
    for(int i = 0; i <= ny; i++){
        delete[] u[i];
        delete[] u_half[i];
    }
    
    delete[] u;
    delete[] u_half;
    // -------------------------
    delete[] F_i;
    delete[] F_j;
    
    delete[] X_i;
    delete[] X_j;
    // -------------------------
}
// END------------------------------------------ FREE MEMORY ------------------------------------------------
