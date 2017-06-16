#ifndef THOMAS_CPP_
#define THOMAS_CPP_

#include <iostream>
#include <cstring>
#include <mpi.h>

#include "thomas_y_helper.h"

using namespace std;

extern int mpirank;
extern int mpisize;

extern bool rank_0;
extern bool rank_np;

extern int my_ny;
extern int nx;

extern int src_up, src_dn;
extern int dst_up, dst_dn;

extern int snd_up_tag, snd_dn_tag;
extern int rcv_up_tag, rcv_dn_tag;

extern MPI_Datatype row_1, column;

void solveMatrix(int n, double *a, double *b, double *c, double *f, double *x){
    double m;
    
    int i_prev;
    
    for(int i = 1; i < n; i++){
        i_prev = i-1;
        
        m = a[i] / b[i_prev];
        
        b[i] -= m * c[i_prev];
        f[i] -= m * f[i_prev];
    }
    
    x[n-1] = f[n-1] / b[n-1];
    
    for(int i = n-2; i >= 0; i--){
        x[i] = (f[i] - c[i] * x[i+1]) / b[i];
    }
}

void down(double** A, double** B, double** C, double** F_i, double** d_l, int i_down){
    double coeff;
    
    int i_next;
    
    for(int i = 0; i <= i_down; i++){
        i_next = i+1;
        // ---------------------------------------
        coeff = B[1][i];
        
        B[1][i] /= coeff;
        C[1][i] /= coeff;
        
        if(!rank_0)
            d_l[1][i] /= coeff;
        
        for(int J = 1; J < nx; J++){
            F_i[J][i] /= coeff;
        }
        // ---------------------------------------
        coeff = A[1][i];
        
        B[1][i_next] -= C[1][i] * coeff;
        A[1][i] -= B[1][i] * coeff;
        
        if(!rank_0)
            d_l[1][i_next] -= d_l[1][i] * coeff;
        
        for(int J = 1; J < nx; J++){
            F_i[J][i_next] -= F_i[J][i] * coeff;
        }
        // ---------------------------------------
    }
    
    coeff = B[1][i_down];
    
    B[1][i_down] /= coeff;
    A[1][i_down] /= coeff;
    C[1][i_down] /= coeff;
    
    if(!rank_0)
        d_l[1][i_down] /= coeff;
    
    for(int J = 1; J < nx; J++){
        F_i[J][i_down] /= coeff;
    }
}

void up(double** A, double** B, double** C, double** F_i, double** d_l, double** d_r, int i_down){
    double coeff;
    
    d_r[1][i_down-1] = C[1][i_down-1];

    C[1][i_down-1] = 0;
    
    int i_next;
    
    for(int i = i_down-2; i >= 0; i--){
        i_next = i+1;
        
        coeff = C[1][i];
        
        C[1][i] -= B[1][i_next] * coeff;
        
        if(!rank_0)
            d_l[1][i] -= d_l[1][i_next] * coeff;
        
        d_r[1][i] -= d_r[1][i_next] * coeff;
        
        for(int J = 1; J < nx; J++){
            F_i[J][i] -= F_i[J][i_next] * coeff;
        }
    }
}

double** thomas_y(double** A_i, double** B_i, double** C_i, double** F_i, double** X_i, Helper* helper){
    // -----------------------------------------
    int M = helper->M;
    
    int H = helper->H;
    int W = helper->W;
    
    int i_0 = helper->i_0;
    int i_down = helper->i_down;
    
    double** d_l = helper->d_l;
    double** d_r = helper->d_r;
    
    helper->flush();
    // -----------------------------------------
    int* block_size = helper->block_size;
    
    int* block_start = helper->block_start;
    int* block_fin = helper->block_fin;
    
    int* block_shift = helper->block_shift;
    
    int block_size_max = helper->block_size_max;
    // -----------------------------------------
    double A[nx][H];
    double C[nx][H];
    double F[nx][H];
    
    double triple_dn[nx][3];
    double triple_up[nx][3];
    
    double sln[mpisize][block_size_max][H];
    // --------------------------------------------
    MPI_Request* req_sln_rcv = helper->req_sln_rcv;
    MPI_Request* req_sln_snd = helper->req_sln_snd;
    
    MPI_Status* stat_sln = helper->stat_sln;
    // --------------------------------------------
    if(!rank_np){
        MPI_Irecv(helper->triple_dn, nx+2, MPI_DOUBLE, src_dn, rcv_dn_tag * 1000, MPI_COMM_WORLD, &(helper->req_triple_down_rcv));
    }
    
    for(int r = 0; r < mpisize; r++){
        MPI_Irecv(sln[r], block_size[r] * H, MPI_DOUBLE, r, (r+1), MPI_COMM_WORLD, &req_sln_rcv[r]);
        
        MPI_Irecv(&A[1][r], 1, row_1, r, r * 3, MPI_COMM_WORLD, &helper->req_diags_rcv[r]);
        MPI_Irecv(&C[1][r], 1, row_1, r, r * 4, MPI_COMM_WORLD, &helper->req_diags_rcv[r + mpisize]);
        MPI_Irecv(&F[1][r], 1, row_1, r, r * 5, MPI_COMM_WORLD, &helper->req_diags_rcv[r + mpisize * 2]);
    }
    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------
    double* a_down_all = helper->a_down_all;
    double* c_down_all = helper->c_down_all;
    double* f_down_all = helper->f_down_all;
    //-----------------------------------------------------------------------------------------------------------------
    d_l[1][0] = A_i[1][0];
    d_r[1][i_down] = B_i[1][i_down];
    //-----------------------------------------------------------------------------------------------------------------
    down(A_i, B_i, C_i, F_i, d_l, i_down);
    up(A_i, B_i, C_i, F_i, d_l, d_r, i_down);
    
    for(int J = 1; J < nx; J++){
        helper->triag_snd_buf[J] = F_i[J][0];
    }
    
    helper->triag_snd_buf[nx] = d_l[1][0];
    helper->triag_snd_buf[nx+1] = d_r[1][0];
    
    if(!rank_0){
        MPI_Isend(helper->triag_snd_buf, nx+2, MPI_DOUBLE, dst_up, snd_up_tag * 1000, MPI_COMM_WORLD, &(helper->req_triags_snd));
    }
    
    for(int J = 1; J < nx; J++){
        for(int i = 0; i < M; i++){
            C_i[J][i] = C_i[1][i];
            
            d_l[J][i] = d_l[1][i];
            d_r[J][i] = d_r[1][i];
        }
    }
    //-----------------------------------------------------------------------------------------------------------------
    if(!rank_np){
        MPI_Wait(&(helper->req_triple_down_rcv), &(helper->stat_triple_down));
    }
    
    for(int J = 1; J < nx; J++){
        double coef;
        // ---------------------------------------
        triple_dn[J][0] = helper->triple_dn[nx];
        triple_dn[J][1] = helper->triple_dn[nx+1];
        triple_dn[J][2] = helper->triple_dn[J];
        // ---------------------------------------
        A_i[J][i_down] = A_i[1][i_down];
        B_i[J][i_down] = B_i[1][i_down];
        
        coef = C_i[J][i_down];
        
        B_i[J][i_down] -= triple_dn[J][0] * coef;
        F_i[J][i_down] -= triple_dn[J][2] * coef;
        
        triple_dn[J][1] *= -coef;
        // ---------------------------------------
        coef = B_i[J][i_down];
        
        A_i[J][i_down] /= coef;
        B_i[J][i_down] /= coef;
        C_i[J][i_down] /= coef;
        F_i[J][i_down] /= coef;
        
        f_down_all[J-1] = F_i[J][i_down];
        
        if(!rank_0){
            d_l[J][i_down] /= coef;
            
            a_down_all[J-1] = d_l[J][i_down];
        }
        
        triple_dn[J][1] /= coef;
        
        c_down_all[J-1] = triple_dn[J][1];
        
        d_r[J][i_down] /= coef;
        // ---------------------------------------
    }
    
    for(int i = 0; i < mpisize; i++){
        MPI_Isend(a_down_all, 1, column, i, mpirank * 3, MPI_COMM_WORLD, &helper->req_diags_snd[i]);
        MPI_Isend(c_down_all, 1, column, i, mpirank * 4, MPI_COMM_WORLD, &helper->req_diags_snd[i + mpisize]);
        MPI_Isend(f_down_all, 1, column, i, mpirank * 5, MPI_COMM_WORLD, &helper->req_diags_snd[i + mpisize * 2]);
    }
    
    MPI_Waitall(mpisize * 3, helper->req_diags_rcv, helper->stat_recv);
    //-----------------------------------------------------------------------------------------------------------------
    for(int J = block_start[mpirank]; J < block_fin[mpirank]; J++){
        for(int i = 0; i < H; i++){
            helper->B[i] = 1;
        }
        
        solveMatrix(H, A[J], helper->B, C[J], F[J], helper->X);
        
        for(int i = 0; i < H; i++){
            sln[mpirank][J - block_shift[mpirank]][i] = helper->X[i];
        }
    }
    
    for(int r = 0; r < mpisize; r++){
        MPI_Isend(sln[mpirank], block_size[mpirank] * H, MPI_DOUBLE, r, (mpirank+1), MPI_COMM_WORLD, &req_sln_snd[r]);
    }
    
    int numdone;
    
    int _mpisize = mpisize;
    
    while(_mpisize--){    
        MPI_Waitany(mpisize, req_sln_rcv, &numdone, stat_sln);
        
        int r = numdone;
        
        for(int J = block_start[r]; J < block_fin[r]; J++){
            if(rank_0){
                X_i[J][i_down] = sln[r][J-block_shift[r]][0];
                
                for(int i = M-2; i >= 0; i--){
                    X_i[J][i] = F_i[J][i] - X_i[J][i_down] * d_r[J][i];
                }
            }else{
                double _l = sln[r][J-block_shift[r]][mpirank-1];
                double _r = sln[r][J-block_shift[r]][mpirank];
                
                X_i[J][i_down] = _r;
                
                for(int i = M-2; i >= 0; i--){
                    X_i[J][i] = F_i[J][i] - (d_l[J][i] * _l + d_r[J][i] * _r);
                }
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------------------      
    return X_i;
}

#endif
