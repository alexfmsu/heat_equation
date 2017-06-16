#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <mpi.h>

#define sqr(x) (x*x)

using namespace std;

// ---------------
char* filename;
// ---------------
double a1, b1;
double a2, b2;

double t_max;

int nx, ny, nt;

double dx, dy, dt;

double* x;
double* y;
double* t;

double** u;
double** u_k;

double _dx, _dy;
double _dx2, _dy2;
// ---------------
int mpirank;
int mpisize;

bool mpiroot;

// BEGIN---------------------------------------- EXACT SOLUTION ---------------------------------------------
double exact(double x, double y, double t){
    return sin(x * t) + cos(y * t);
}
// END------------------------------------------ EXACT SOLUTION ---------------------------------------------

// BEGIN---------------------------------------- F(x, y, t) -------------------------------------------------
double F(double x, double y, double t){
    return (x * cos(x * t) - y * sin(y * t)) * (1 - 2 * t)  + sqr(t) * (sqr(x) + sqr(y)) * (sin(x * t) + cos(y * t));
}
// END------------------------------------------ F(x, y, t) -------------------------------------------------

// BEGIN---------------------------------------- K(x, y) ----------------------------------------------------
double K(double x, double y){
    return sqr(x) + sqr(y);
}

double K_ij(int i, int j){
    return K(x[j], y[i]);
}

double** K_i_plus;
double** K_i_minus;

double** K_j_plus;
double** K_j_minus;

void init_K(){
    try{
        K_j_plus = new double*[ny+1];
        K_j_minus = new double*[ny+1];
        
        K_i_plus = new double*[ny+1];
        K_i_minus = new double*[ny+1];
        
        for(int i = 0; i <= ny; i++){
            K_j_plus[i] = new double[nx+1];
            K_j_minus[i] = new double[nx+1];
            
            K_i_plus[i] = new double[nx+1];
            K_i_minus[i] = new double[nx+1];
        }
    }catch(std::bad_alloc){
        cout << "Cannot allocate memory" << endl;
        
        exit(1);
    }
    
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            K_j_plus[i][j] = 2 * K_ij(i, j) * K_ij(i, j+1) / (K_ij(i, j) + K_ij(i, j+1));
            K_j_minus[i][j] = 2 * K_ij(i, j) * K_ij(i, j-1) / (K_ij(i, j) + K_ij(i, j-1));
            
            K_i_plus[i][j] = 2 * K_ij(i, j) * K_ij(i+1, j) / (K_ij(i, j) + K_ij(i+1, j));
            K_i_minus[i][j] = 2 * K_ij(i, j) * K_ij(i-1, j) / (K_ij(i, j) + K_ij(i-1, j));
        }
    }
}
// END------------------------------------------ K(x, y) ----------------------------------------------------

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
        cout << "         10 10 1000" << endl;
        cout << endl;
        
        exit(1);
    }
    
    fclose(f);
}
// END------------------------------------------ READ DATA --------------------------------------------------

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
    
    _dy = 1. / dy;
    _dx = 1. / dx;
    
    _dy2 = sqr(_dy);
    _dx2 = sqr(_dx);
    
    try{
        y = new double[ny+1];
        x = new double[nx+1];
        t = new double[nt+1];
        
        u = new double*[ny+1];
        u_k = new double*[ny+1];
        
        for(int i = 0; i <= ny; i++){
            u[i] = new double[nx+1];
            u_k[i] = new double[nx+1];
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
    
    init_K();
}
// END------------------------------------------ SET GRID ---------------------------------------------------

// BEGIN---------------------------------------- INITIAL CONDITIONS -----------------------------------------
double g_0(int i, int j){
    return exact(x[j], y[i], t[0]);
}

void set_initials(){
    for(int i = 1; i < ny; i++){
        for(int j = 1; j < nx; j++){
            u[i][j] = g_0(i, j);
        }
    }
}
// END------------------------------------------ INITIAL CONDITIONS -----------------------------------------

// BEGIN---------------------------------------- BOUNDARY CONDITIONS ----------------------------------------
double g_11(int i, int k){
    return exact(x[0], y[i], t[k]);
}

double g_12(int i, int k){
    return exact(x[nx], y[i], t[k]);
}

double g_21(int j, int k){
    return exact(x[j], y[0], t[k]);
}

double g_22(int j, int k){
    return exact(x[j], y[ny], t[k]);
}

void set_boundaries(int k){
    for(int i = 0; i <= ny; i++){
        u[i][0] = g_11(i, k);
        u[i][nx] = g_12(i, k);
    }
    
    for(int j = 0; j <= nx; j++){
        u[0][j] = g_21(j, k);
        u[ny][j] = g_22(j, k);
    }
}
// END------------------------------------------ BOUNDARY CONDITIONS ----------------------------------------

// BEGIN---------------------------------------- CHECK COURANT CONDITION ------------------------------------
void check_courant(){
    double max_k = 0;
    
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            max_k = max(max_k, K_ij(i, j));
        }
    }
    
    double h = min(dx, dy);
    
    if(dt >= (1./4) * sqr(h) / max_k){
        cout << endl << "Current condition is not implemented" << endl << endl;
        
        exit(1);
    }
}
// END------------------------------------------ CHECK COURANT CONDITION ------------------------------------

// BEGIN---------------------------------------- SOLVE ------------------------------------------------------
void solve(){
    set_initials();
    set_boundaries(0);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();    
    
    for(int k = 1; k <= nt; k++){
        for(int i = 1; i < ny; i++){
            for(int j = 1; j < nx; j++){
                double d_i_plus = u[i+1][j] - u[i][j];
                double d_i_minus = u[i][j] - u[i-1][j];
                
                double d_j_plus = u[i][j+1] - u[i][j];
                double d_j_minus = u[i][j] - u[i][j-1];
                
                u_k[i][j] = u[i][j] + dt * (
                    _dy2 * (K_i_plus[i][j] * d_i_plus - K_i_minus[i][j] * d_i_minus)
                    + 
                    _dx2 * (K_j_plus[i][j] * d_j_plus - K_j_minus[i][j] * d_j_minus) 
                    +
                    F(x[j], y[i], t[k-1])
                );
            }
        }
        
        swap(u_k, u);
        
        set_boundaries(k);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();    
    
    if(mpiroot){
        cout << "Time: " << (t2 - t1) << endl << endl;
    }
}
// END------------------------------------------ SOLVE ------------------------------------------------------

// -----------------------
void print_grid();

void print_y();
void print_x();
void print_t();

void print_u();
void print_u_exact(int k);
void print_error();

void free_memory();
// -----------------------

int main(int argc, char** argv){
    cout << fixed << setprecision(10);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    
    mpiroot = (mpirank == 0);
    
    if(mpiroot){
        if(argc < 2){
            cout << "Usage: ./a.out data" << endl;
            
            exit(1);
        }
    }
    
    filename = argv[1];
    
    set_grid();
    
    check_courant();
    
    solve();
    
    print_error();
    
    free_memory();
    
    MPI_Finalize();

    return 0;
}

// BEGIN---------------------------------------- PRINT GRID -------------------------------------------------
void print_y(){
    cout << "y: ";
    
    for(int i = 0; i <= ny; i++){
        cout << y[i] << " ";
    }
    
    cout << endl << endl;
}

void print_x(){
    cout << "x: ";
    
    for(int i = 0; i <= nx; i++){
        cout << x[i] << " ";
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
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            cout << u[i][j] << " ";
        }
        
        cout << endl;
    }
    
    cout << endl;
}

void print_u_exact(int k){
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            cout << exact(x[j], y[i], t[k]) << " ";
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
    
    for(int i = 0; i <= ny; i++){
        for(int j = 0; j <= nx; j++){
            double sln = u[i][j];
            
            double sln_e = exact(x[j], y[i], t[nt]);
            
            if(sln_e == 0){
                continue;
            }
            
            double abs_error = abs(sln - sln_e);
            double rel_error = abs_error / abs(sln_e);
            
            max_abs_error = max(max_abs_error, abs_error);
            max_rel_error = max(max_rel_error, rel_error);
        }
    }
    
    cout << "Abs. error: " << max_abs_error << endl;
    cout << "Rel. error: " << max_rel_error << endl;
}
// END------------------------------------------ PRINT ERROR ------------------------------------------------

// BEGIN---------------------------------------- FREE MEMORY ------------------------------------------------
void free_memory(){
    delete[] y;
    delete[] x;
    delete[] t;
    
    for(int i = 0; i <= ny; i++){
        delete[] K_j_plus[i];
        delete[] K_j_minus[i];
        delete[] K_i_plus[i];
        delete[] K_i_minus[i];
        
        delete[] u[i];
        delete[] u_k[i];
    }
    
    delete[] K_j_plus;
    delete[] K_j_minus;
    delete[] K_i_plus;
    delete[] K_i_minus;
    
    delete[] u;
    delete[] u_k;
}
// END------------------------------------------ FREE MEMORY ------------------------------------------------
