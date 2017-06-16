#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#include "thomas_x.cpp"
#include "thomas_y.cpp"
#include "thomas_y_helper.h"

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
int mpirank;
int mpisize;

bool rank_0;
bool rank_np;

int src_up, src_dn;
int dst_up, dst_dn;

int snd_up_tag, snd_dn_tag;
int rcv_up_tag, rcv_dn_tag;

MPI_Datatype row, row_1, column;
// ------------------------
int ny_per_node;
int ny_tail;

int my_ny;
    
int row_0_ny;
int row_n_ny;

double** u_up;
double** u_down;

double** u_left;
double** u_right;
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
    // --------------------------------------------------
    ny_per_node = ((ny + 1) + 2 * (mpisize-1)) / mpisize;
    
    ny_tail = ((ny + 1) + 2 * (mpisize-1)) % mpisize;

    my_ny = ny_per_node - 1;
    
    if(mpirank < ny_tail)
        my_ny++;
    // --------------------------------------------------
    try{
        x = new double[nx+1];
        y = new double[my_ny+1];
        t = new double[nt+1];
        
        u = new double*[my_ny+1];
        u_half = new double*[my_ny+1];
        
        for(int i = 0; i <= my_ny; i++){
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
    // --------------------------------------------------
    double y0 = 0;
    
    if(mpisize == 1){
        y0 = a2;
    }else if(!rank_0){
        int _y0 = 0;
        
        if(ny_tail){
            _y0 = (mpirank <= ny_tail) ? mpirank : ny_tail;
        }
        
        y0 = a2 + ((ny_per_node-2) * (mpirank) + _y0) * dy;
    }
    
    for(int i = 0; i <= my_ny; i++){
        y[i] = y0 + i * dy;
    }
    // --------------------------------------------------
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
    int i_1 = (rank_0) ? 0 : 1;
    int i_2 = (rank_np) ? (my_ny) : (my_ny-1);

    for(int i = i_1; i <= i_2; i++){
        for(int j = 0; j <= nx; j++){
            u[i][j] = g_0(j, i);
        }
    }
}
// END------------------------------------------ INITIAL CONDITIONS -----------------------------------------

// BEGIN---------------------------------------- BOUNDARY CONDITIONS ----------------------------------------
void set_boundaries(double** _u, int k){
    if(!rank_0){
        for(int j = 0; j <= nx; j++){
            _u[my_ny][j] = u_down[k][j];
        }
    }
    
    if(!rank_np){
        for(int j = 0; j <= nx; j++){
            _u[0][j] = u_up[k][j];
        }    
    }
    
    for(int i = 0; i <= my_ny; i++){
        _u[i][0] = u_left[k][i];
        _u[i][nx] = u_right[k][i];
    }
}
// END------------------------------------------ BOUNDARY CONDITIONS ----------------------------------------

// BEGIN---------------------------------------- MPI_PREPARE ------------------------------------------------
void mpi_prepare(){
    MPI_Type_contiguous(nx, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    
    MPI_Type_contiguous(nx-1, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    
    MPI_Type_vector(nx-1, 1, mpisize, MPI_DOUBLE, &row_1);
    MPI_Type_commit(&row_1);
    
    src_dn = dst_dn = mpirank+1;
    src_up = dst_up = mpirank-1;
    
    snd_up_tag = 1000 * dst_up  + 2;
    snd_dn_tag = 1000 * dst_dn  + 0;
    
    rcv_up_tag = 1000 * mpirank  + 0;
    rcv_dn_tag = 1000 * mpirank  + 2;
}
// END------------------------------------------ MPI_PREPARE ------------------------------------------------

// BEGIN---------------------------------------- PREPARE_BOUNDARIES -----------------------------------------
void prepare_boundaries(){
    u_up = new double*[(nt+1)*2];
    u_down = new double*[(nt+1)*2];
    u_left = new double*[(nt+1)*2];
    u_right = new double*[(nt+1)*2];
    
    for(int k = 0; k <= nt*2; k++){
        u_up[k] = new double[nx+1];
        u_down[k] = new double[nx+1];

        u_left[k] = new double[my_ny+1];
        u_right[k] = new double[my_ny+1];
    }
    
    for(int k = 0; k <= nt; k++){
        for(int j = 0; j < nx; j++){
            u_up[k*2][j] = u_exact(x[j], y[0], t[k]);
            u_down[k*2][j] = u_exact(x[j], y[my_ny], t[k]);
            
            if(k < nt){
                u_up[k*2+1][j] = u_exact(x[j], y[0], t[k]+dt_1_2);
                u_down[k*2+1][j] = u_exact(x[j], y[my_ny], t[k]+dt_1_2);
            }
        }
        
        for(int i = 0; i <= my_ny; i++){
            u_left[k*2][i] = u_exact(x[0], y[i], t[k]);
            u_right[k*2][i] = u_exact(x[nx], y[i], t[k]);
            
            if(k < nt){
                u_left[k*2+1][i] = u_exact(x[0], y[i], t[k]+dt_1_2);
                u_right[k*2+1][i] = u_exact(x[nx], y[i], t[k]+dt_1_2);
            }
        }
    }
}
// END------------------------------------------ PREPARE_BOUNDARIES -----------------------------------------

// BEGIN---------------------------------------- SOLVE ------------------------------------------------------
void solve(){
    mpi_prepare();
    // ---------------------------
    set_triag_coeff();
    // ---------------------------
    double* F_j = new double[nx-1];
    
    double* X_j = new double[nx-1];
    // ---------------------------
    prepare_boundaries();
    // --------------------------------------------------------
    set_initials();
    // --------------------------------------------------------
    double** A_nx_i = new double*[nx];
    double** B_nx_i = new double*[nx];
    double** C_nx_i = new double*[nx];
    
    double** F_nx_i = new double*[nx];
    
    double** X_nx_i = new double*[nx];
    
    for(int i = 0; i < nx; i++){
        A_nx_i[i] = new double[my_ny];
        B_nx_i[i] = new double[my_ny];
        C_nx_i[i] = new double[my_ny];
        
        F_nx_i[i] = new double[my_ny];
        
        X_nx_i[i] = new double[my_ny+1];
    }
    // --------------------------------------------------------
    Helper* helper = new Helper(mpirank, mpisize, nx, my_ny, int(y[0] / dy));
    // --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    
    for(int k = 0; k < nt; k++){
        double t_half = t[k] + dt_1_2;
        double t_k = t_half + dt_1_2;
        
        // BEGIN-------------------------------- x-direction ------------------------------------------------
        MPI_Status st1, st2;

        MPI_Request req_snd_up, req_snd_dn;
        MPI_Request req_rcv_up, req_rcv_dn;
        
        if(mpisize != 1){    
            if(!rank_0){
                MPI_Irecv(&u[0][1], 1, row, src_up, rcv_up_tag, MPI_COMM_WORLD, &req_rcv_up);
            }
            
            if(!rank_np){
                MPI_Irecv(&u[my_ny][1], 1, row, src_dn, rcv_dn_tag, MPI_COMM_WORLD, &req_rcv_dn);
            }
        }
        
        set_boundaries(u_half, 2*k+1);
        
        if(mpisize != 1){    
            if(!rank_0){
                MPI_Isend(&u[1][1], 1, row,   dst_up, snd_up_tag, MPI_COMM_WORLD, &req_snd_dn);
            }
            
            if(!rank_np){
                MPI_Isend(&u[my_ny-1][1], 1, row,   dst_dn, snd_dn_tag, MPI_COMM_WORLD, &req_snd_up);
            }
        }
        
        for(int i = 2; i < my_ny-1; i++){
            u_half = thomas_x(u, A_j, B_j, C_j, F_j, X_j, k, i, t_half);
        }
        
        if(!rank_0){    
            MPI_Wait(&req_rcv_up, &st1);
        }
        
        u_half = thomas_x(u, A_j, B_j, C_j, F_j, X_j, k, 1, t_half);
        
        if(!rank_np){    
            MPI_Wait(&req_rcv_dn, &st2);    
        }
        
        u_half = thomas_x(u, A_j, B_j, C_j, F_j, X_j, k, my_ny-1, t_half);
        // END---------------------------------- x-direction ------------------------------------------------
        
        // BEGIN-------------------------------- y-direction ------------------------------------------------
        set_boundaries(u, k*2+2);
        
        for(int _i = 0; _i < my_ny; _i++){
            A_nx_i[1][_i] = A_i[_i];
            B_nx_i[1][_i] = B_i[_i];
            C_nx_i[1][_i] = C_i[_i];
        }
        
        for(int j = 1; j < nx; j++){
            for(int i = 1; i < my_ny; i++){
                F_nx_i[j][i-1] = (u_half[i][j-1] - 2 * u_half[i][j] + u_half[i][j+1]) / dx2 + F(x[j], y[i], t_half) + 2 * u_half[i][j] / dt;               
            }
            
            if(rank_0){
                F_nx_i[j][0] += u_up[2*k+2][j] / dy2;
            }else if(rank_np){
                F_nx_i[j][my_ny-2] += u_down[2*k+2][j] / dy2;
            }
        }
        
        X_nx_i = thomas_y(A_nx_i, B_nx_i, C_nx_i, F_nx_i, X_nx_i, helper); 
        
        for(int j = 1; j < nx; j++){    
            for(int i = 0; i < my_ny-1; i++){
                u[i+1][j] = X_nx_i[j][i];
            }
        }
        // END---------------------------------- y-direction ------------------------------------------------
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    
    if(rank_0){
        cout << "nprocs: " << mpisize << endl;
        cout << "Time: " << (t2 - t1) << endl << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}
// END------------------------------------------ SOLVE ------------------------------------------------------



int main(int argc, char** argv){
    cout << fixed << setprecision(8);
    
    if(argc < 2){
        cout << "Usage: ./a.out data" << endl;
        
        exit(1);
    }
    
    filename = argv[1];
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    
    rank_0 = (mpirank == 0);
    rank_np = (mpirank == mpisize-1);
    
    set_grid();
    
    solve();
    
    print_error();
    
    free_memory();
    
    MPI_Finalize();
    
    return 0;
}



// BEGIN---------------------------------------- PRINT GRID -------------------------------------------------
void print_x(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank_0){
        cout << "x: ";
        
        for(int j = 0; j <= nx; j++){
            cout << x[j] << " ";
        }
        
        cout << endl << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_y(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            cout << "my_ny: " << my_ny << endl;
            cout << "y: ";
            
            for(int i = 0; i <= my_ny; i++){
                cout << y[i] << " ";
            }
            
            cout << endl << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_t(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank_0){
        cout << "t: ";
        
        for(int k = 0; k <= nt; k++){
            cout << t[k] << " ";
        }
        
        cout << endl << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_grid(){
    print_x();
    print_y();
    print_t();
}
// END------------------------------------------ PRINT GRID -------------------------------------------------

// BEGIN---------------------------------------- PRINT SOLUTION ---------------------------------------------
void print_u(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(mpirank == 0){
        cout << "u:" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            for(int i = 0; i <= my_ny; i++){
                for(int j = 0; j <= nx; j++){
                    cout << setw(12) << setprecision(8) << u[i][j] << " ";
                }
                
                cout << endl;
            }
            
            cout << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_u_half(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    cout << setprecision(8);

    if(mpirank == 0){
        cout << "u_half:" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            for(int i = 0; i <= my_ny; i++){
                for(int j = 0; j <= nx; j++){
                    cout << setw(12) << setprecision(9) << u_half[i][j] << " ";
                }
                
                cout << endl;
            }
            
            cout << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_u_exact(int k){
    MPI_Barrier(MPI_COMM_WORLD);
    
    cout << setprecision(8);
    
    if(mpirank == 0){
        cout << "u_half:" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            for(int i = 0; i <= my_ny; i++){
                for(int j = 0; j <= nx; j++){
                    cout << setw(12) << setprecision(9) << u_exact(x[j], y[i], t[k]) << " ";
                }
                
                cout << endl;
            }
            
            cout << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}
// END------------------------------------------ PRINT SOLUTION ---------------------------------------------

// BEGIN---------------------------------------- PRINT ERROR ------------------------------------------------
void print_error(){
    double max_abs_error = 0;
    double max_rel_error = 0;
      
    int i1, i2;
    int j1 = 0, j2 = nx;
    
    if(mpirank == 0){
        i1 = 0;
        i2 = my_ny-1;
    }else if(mpirank == mpisize-1){
        i1 = 1;
        i2 = my_ny;
    }else{
        i1 = 1;
        i2 = my_ny-1;
    }
    
    for(int i = i1; i <= i2; i++){
        for(int j = j1; j <= j2; j++){
            double sln = u[i][j];
            
            double sln_e = u_exact(x[j], y[i], t[nt]);
            
            if(abs(sln_e) < 1e-10 || abs(sln) < 1e-9){
                continue;
            }
            
            double abs_error = abs(sln - sln_e);
            double rel_error = abs_error / abs(sln_e);
            
            max_abs_error = max(max_abs_error, abs_error);
            max_rel_error = max(max_rel_error, rel_error);
        }
    }
    
    double* abs_err = new double[mpisize];
    double* rel_err = new double[mpisize];
    
    MPI_Gather(&max_abs_error, 1, MPI_DOUBLE, abs_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&max_rel_error, 1, MPI_DOUBLE, rel_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpirank == 0){
        double max_abs_error = 0;
        double max_rel_error = 0;
        
        for(int i = 0; i < mpisize; i++){
            max_abs_error = max(abs_err[i], max_abs_error);
            max_rel_error = max(rel_err[i], max_rel_error);
        }
        
        cout << "Abs. error: " << setprecision(8) << max_abs_error << endl;
        cout << "Rel. error: " << setprecision(8) << max_rel_error << endl;
    }
}
// END------------------------------------------ PRINT ERROR ------------------------------------------------

// BEGIN---------------------------------------- FREE MEMORY ------------------------------------------------
void free_memory(){
    delete[] x;
    delete[] y;
    delete[] t;
    
    for(int i = 0; i <= my_ny; i++){
        delete[] u[i];
        delete[] u_half[i];
    }
    
    delete[] u;
    delete[] u_half;
}
// END------------------------------------------ FREE MEMORY ------------------------------------------------
