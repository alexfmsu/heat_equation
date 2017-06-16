#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <map>

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
bool mpiroot;

int mpirank;
int mpisize;

int nprows;
int npcols;

int my_row;
int my_col;

int my_nx;
int my_ny;

int ny_per_node;
int nx_per_node;
// ---------------

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
        K_j_plus = new double*[my_ny+1];
        K_j_minus = new double*[my_ny+1];
        
        K_i_plus = new double*[my_ny+1];
        K_i_minus = new double*[my_ny+1];
        
        for(int i = 0; i <= my_ny; i++){
            K_j_plus[i] = new double[my_nx+1];
            K_j_minus[i] = new double[my_nx+1];
            
            K_i_plus[i] = new double[my_nx+1];
            K_i_minus[i] = new double[my_nx+1];
        }
    }catch(std::bad_alloc){
        cout << "Cannot allocate memory" << endl;
        
        exit(1);
    }
    
    for(int i = 0; i <= my_ny; i++){
        for(int j = 0; j <= my_nx; j++){
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

// BEGIN---------------------------------------- SET PROC GRID ----------------------------------------------
void set_proc_grid(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    nprows = sqrt(mpisize);
    
    while(mpisize % nprows){
        nprows--;
    }
    
    npcols = mpisize / nprows;
    
    my_row = mpirank / npcols;
    my_col = mpirank - my_row * npcols;
    
    MPI_Barrier(MPI_COMM_WORLD);
}
// END------------------------------------------ SET PROC GRID ----------------------------------------------

// BEGIN---------------------------------------- SET GRID ---------------------------------------------------
void set_grid(){
    if(mpiroot){
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
    }
    
    MPI_Bcast(&a1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    _dy = 1. / dy;
    _dx = 1. / dx;
    
    _dy2 = sqr(_dy);
    _dx2 = sqr(_dx);
    
    set_proc_grid();
    
    ny_per_node = (ny + 1) / nprows;
    nx_per_node = (nx + 1) / npcols;
    
    int ny_tail = (ny + 1) % nprows;
    int nx_tail = (nx + 1) % npcols;
    
    my_ny = ny_per_node;
    my_nx = nx_per_node;
    
    if(nprows == 1){
        my_ny = ny;
    }else if(my_row == 0){
        my_ny += ny_tail;
    }else if(my_row != nprows-1){
        my_ny++;
    }
    
    if(npcols == 1){
        my_nx = nx;
    }else if(my_col == 0){
        my_nx += nx_tail;
    }else if(my_col != npcols-1){
        my_nx++;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            if(my_col != 0 && my_col != npcols-1 && my_nx < 2 
               ||
               my_row != 0 && my_row != nprows-1 && my_ny < 2
            ){
                cout << "Set another process grid" << endl;
                
                exit(1); 
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double x0, y0;
    
    if(my_row == 0){
        y0 = a2;
    }else{
        y0 = a2 + (ny_tail + ny_per_node * my_row - 1) * dy;
    }
    
    if(my_col == 0){
        x0 = a1;
    }else{
        x0 = a1 + (nx_tail + nx_per_node * my_col - 1) * dx;
    }
    
    try{
        y = new double[my_ny+1];
        x = new double[my_nx+1];
        t = new double[nt+1];
        
        u = new double*[my_ny+1];
        u_k = new double*[my_ny+1];
        
        for(int i = 0; i <= my_ny; i++){
            u[i] = new double[my_nx+1];
            u_k[i] = new double[my_nx+1];
        }
    }catch(std::bad_alloc){
        cout << "Cannot allocate memory" << endl;
        
        exit(1);
    }
    
    for(int i = 0; i <= my_ny; i++){
        y[i] = y0 + i * dy;
    }
    
    for(int j = 0; j <= my_nx; j++){
        x[j] = x0 + j * dx;
    }
    
    for(int k = 0; k <= nt; k++){
        t[k] = k * dt;
    }
    
    init_K();
}
// END------------------------------------------ SET GRID ---------------------------------------------------

// BEGIN---------------------------------------- BOUNDARY CONDITIONS ----------------------------------------
double g_0(int i, int j){
    return exact(x[j],  y[i], t[0]);
}

double g_11(int i, int k){
    return exact(x[0], y[i], t[k]);
}

double g_12(int i, int k){
    return exact(x[my_nx], y[i], t[k]);
}

double g_21(int j, int k){
    return exact(x[j], y[0], t[k]);
}

double g_22(int j, int k){
    return exact(x[j], y[my_ny], t[k]);
}

void set_initials(){
    for(int i = 1; i < my_ny; i++){
        for(int j = 1; j < my_nx; j++){
            u[i][j] = g_0(i, j);
        }
    }
}

void set_boundaries(int k){
    if(my_row == 0){
        for(int j = 0; j <= my_nx; j++){
            u[0][j] = g_21(j, k);
        }
    }
    
    if(my_row == nprows - 1){
        for(int j = 0; j <= my_nx; j++){
            u[my_ny][j] = g_22(j, k);
        }
    }
    
    if(my_col == 0){
        for(int i = 0; i <= my_ny; i++){
            u[i][0] = g_11(i, k);
        }
    }
    
    if(my_col == npcols - 1){
        for(int i = 0; i <= my_ny; i++){
            u[i][my_nx] = g_12(i, k);
        }
    }
}
// END------------------------------------------ BOUNDARY CONDITIONS ----------------------------------------

// BEGIN---------------------------------------- CHECK COURANT CONDITION ------------------------------------
void check_courant(){
    double my_max_k = 0;
    
    for(int i = 0; i <= my_ny; i++){
        for(int j = 0; j <= my_nx; j++){
            my_max_k = max(my_max_k, K_ij(i, j));
        }
    }
    
    double max_k[mpisize];

    MPI_Gather(&my_max_k, 1, MPI_DOUBLE, max_k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpiroot){
        double MAX_K = 0;
        
        for(int i = 0; i < mpisize; i++){
            MAX_K = max(MAX_K, max_k[i]);
        }
        
        double h = min(dx, dy);
        
        if(dt >= (1./4) * sqr(h) / MAX_K){
            cout << endl << "Current condition is not implemented" << endl << endl;
            
            exit(1);
        }
    }
}
// END------------------------------------------ CHECK COURANT CONDITION ------------------------------------

// BEGIN---------------------------------------- SOLVE ------------------------------------------------------
void solve(){
    set_boundaries(0);
    set_initials();
    
    MPI_Datatype row;
    MPI_Type_contiguous(my_nx-1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    
    MPI_Datatype column; 
    MPI_Type_contiguous(my_ny-1, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    
    int dest_up = mpirank - npcols;
    int dest_down = mpirank + npcols;
    int dest_left = mpirank - 1;
    int dest_right = mpirank + 1;
    
    int source_up = dest_up;
    int source_down = dest_down;
    int source_left = dest_left;
    int source_right = dest_right;
    
    int snd_up_tag = 1000 * dest_up  + 2;
    int snd_dn_tag = 1000 * dest_down  + 0;
    
    int snd_left_tag = 1000 * dest_left  + 1;
    int snd_right_tag = 1000 * dest_right + 3;
    
    int rcv_up_tag = 1000 * mpirank  + 0;
    int rcv_dn_tag = 1000 * mpirank  + 2;

    int rcv_left_tag = 1000 * mpirank  + 3;
    int rcv_right_tag = 1000 * mpirank  + 1;

    // map<string, MPI_Status*> st = {
    //     {"up", new MPI_Status},
    //     {"down", new MPI_Status},
    //     {"left", new MPI_Status},
    //     {"right", new MPI_Status}
    // };
    
    // map<string, int> send_tag = {
    //     {    "up", 1000 * dest_up    + 2 },
    //     {  "down", 1000 * dest_down  + 0 },
        
    //     {  "left", 1000 * dest_left  + 1 },
    //     { "right", 1000 * dest_right + 3 },
    // };
    
    // map<string, int> recv_tag = {
    //     {    "up", 1000 * mpirank + 0 },
    //     {  "down", 1000 * mpirank + 2 },
        
    //     {  "left", 1000 * mpirank + 3 },
    //     { "right", 1000 * mpirank + 1 },
    // };
    
    vector<double> left_sd;
    vector<double> right_sd;
    
    double left[ny_per_node];
    double right[ny_per_node];
    
    MPI_Status st_rcv_up;
    MPI_Status st_rcv_down;

    MPI_Status st_rcv_left;
    MPI_Status st_rcv_right;

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();    
    
    for(int k = 1; k <= nt; k++){
        left_sd.clear();
        right_sd.clear();
        
        for(int i = 1; i < my_ny; i++){
            left_sd.push_back(u[i][1]);
            right_sd.push_back(u[i][my_nx-1]);
        }
        
        if(npcols > 1){
            if(my_col == 0){
                MPI_Send(right_sd.data(), 1, column,   dest_right, snd_right_tag, MPI_COMM_WORLD);
                MPI_Recv(          right, 1, column, source_right, rcv_right_tag, MPI_COMM_WORLD, &st_rcv_right);
                
                for(int i = 0; i < my_ny; i++){
                    u[i+1][my_nx] = right[i];
                }
            }else if(my_col == npcols-1){
                MPI_Send(left_sd.data(), 1, column,   dest_left, snd_left_tag, MPI_COMM_WORLD);
                MPI_Recv(          left, 1, column, source_left, rcv_left_tag, MPI_COMM_WORLD, &st_rcv_left);
                
                for(int i = 0; i < my_ny; i++){
                    u[i+1][0] = left[i];
                }
            }else{
                MPI_Send( left_sd.data(), 1, column,  dest_left,  snd_left_tag, MPI_COMM_WORLD);
                MPI_Send(right_sd.data(), 1, column, dest_right, snd_right_tag, MPI_COMM_WORLD);
                
                MPI_Recv( left, 1, column,  source_left,  rcv_left_tag, MPI_COMM_WORLD, &st_rcv_right);
                MPI_Recv(right, 1, column, source_right, rcv_right_tag, MPI_COMM_WORLD,  &st_rcv_left);
                
                for(int i = 0; i < my_ny; i++){
                    u[i+1][0] = left[i];
                    u[i+1][my_nx] = right[i];
                }
            }
        }
        
        if(nprows > 1){
            if(my_row == 0){
                MPI_Send(&u[my_ny-1][1], 1, row,   dest_down, snd_dn_tag, MPI_COMM_WORLD);
                MPI_Recv(  &u[my_ny][1], 1, row, source_down, rcv_dn_tag, MPI_COMM_WORLD, &st_rcv_down);
            }else if(my_row == nprows - 1){
                MPI_Send(&u[1][1], 1, row,   dest_up, snd_up_tag, MPI_COMM_WORLD);
                MPI_Recv(&u[0][1], 1, row, source_up, rcv_up_tag, MPI_COMM_WORLD, &st_rcv_up);
            }else{
                MPI_Send(      &u[1][1], 1, row,     dest_up,   snd_up_tag, MPI_COMM_WORLD);
                MPI_Send(&u[my_ny-1][1], 1, row, source_down, snd_dn_tag, MPI_COMM_WORLD);
                
                MPI_Recv(    &u[0][1], 1, row,     dest_up,   rcv_up_tag, MPI_COMM_WORLD,   &st_rcv_up);
                MPI_Recv(&u[my_ny][1], 1, row, source_down, rcv_dn_tag, MPI_COMM_WORLD, &st_rcv_down);
            }
        }
        
        #pragma omp parallel for
        for(int i = 1; i < my_ny; i++){
            for(int j = 1; j < my_nx; j++){
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
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            
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

void print_x(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            
            cout << "x: ";
            cout << nx_per_node << endl;
            for(int i = 0; i <= my_nx; i++){
                cout << x[i] << " ";
            }
            
            cout << endl << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_t(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            
            cout << "t: ";
            
            for(int k = 0; k <= nt; k++){
                cout << t[k] << " ";
            }
            
            cout << endl << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_grid(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;

            cout << "x: ";
            
            for(int i = 0; i <= my_nx; i++){
                cout << x[i] << " ";
            }
            
            cout << endl << endl;

            cout << "y: ";
            
            for(int i = 0; i <= my_ny; i++){
                cout << y[i] << " ";
            }
            
            cout << endl << endl;

            cout << "t: ";
            
            for(int k = 0; k <= nt; k++){
                cout << t[k] << " ";
            }
            
            cout << endl << endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);    
}
// END------------------------------------------ PRINT GRID -------------------------------------------------

// BEGIN---------------------------------------- PRINT SOLUTION ---------------------------------------------
void print_u(){
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            
            for(int i = 0; i <= my_ny; i++){
                for(int j = 0; j <= my_nx; j++){
                    cout << u[i][j] << " ";
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
    
    for(int r = 0; r < mpisize; r++){
        if(r == mpirank){
            cout << "rank: " << r << endl;
            
            for(int i = 0; i <= my_ny; i++){
                for(int j = 0; j <= my_nx; j++){
                    cout << exact(x[j], y[i], t[k]) << " ";
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
    
    int i1 = 1, i2 = my_ny;
    int j1 = 1, j2 = my_nx;
    
    if(my_row == 0){
        i1--;
    }else if(my_row == nprows-1){
        i2++;
    }
    
    if(my_col == 0){
        j1--;
    }else if(my_col == npcols-1){
        j2++;
    }
    
    for(int i = i1; i < i2; i++){
        for(int j = j1; j < j2; j++){
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
    
    double* abs_err = new double[mpisize];
    double* rel_err = new double[mpisize];
    
    MPI_Gather(&max_abs_error, 1, MPI_DOUBLE, abs_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&max_rel_error, 1, MPI_DOUBLE, rel_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpiroot){
        double max_abs_error = 0;
        double max_rel_error = 0;
        
        for(int i = 0; i < mpisize; i++){
            max_abs_error = max(abs_err[i], max_abs_error);
            max_rel_error = max(rel_err[i], max_rel_error);
        }
        
        cout << "Abs. error: " << max_abs_error << endl;
        cout << "Rel. error: " << max_rel_error << endl;
    }
}
// END------------------------------------------ PRINT ERROR ------------------------------------------------

// BEGIN---------------------------------------- FREE MEMORY ------------------------------------------------
void free_memory(){
    delete[] y;
    delete[] x;
    delete[] t;
    
    for(int i = 0; i <= my_ny; i++){
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
