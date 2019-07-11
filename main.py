import numpy as np
import mayavi.mlab as mlab
import scipy.io as scio
from math import atan2
import os
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from scipy import sparse
import time
import pickle
from scipy.spatial import KDTree


colorMaps = ["Reds", "Oranges", "Purples", "Accent", "black-white", "blue-red",
             "Blues", "bone", "Greens", "Greys", "purples"]


def save_pickle_file(filename, file):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)
        print("save {}".format(filename))


def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            file = pickle.load(f)
        return file
    else:
        print("{} not exist".format(filename))


def loadObj(path):
    """Load obj file
    读取三角形和四边形的mesh
    返回vertex和face的list
    """
    if path.endswith('.obj'):
        f = open(path, 'r')
        lines = f.readlines()
        vertics = []
        faces = []
        for line in lines:
            if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                line_split = line.split()
                ver = line_split[1:4]
                ver = [float(v) for v in ver]
                # print(ver)
                vertics.append(ver)
            else:
                if line.startswith('f'):
                    line_split = line.split()
                    if '/' in line:
                        tmp_faces = line_split[1:]
                        f = []
                        for tmp_face in tmp_faces:
                            f.append(int(tmp_face.split('/')[0]))
                        faces.append(f)
                    else:
                        face = line_split[1:]
                        face = [int(fa) for fa in face]
                        faces.append(face)
        return np.array(vertics, dtype=np.float32), np.array(faces, dtype=np.int32)

    else:
        print('格式不正确，请检查obj格式')
        return


def writeObj(file_name_path, vertexs, faces):
    """write the obj file to the specific path
       file_name_path:保存的文件路径
       vertexs:顶点数组 list
       faces: 面 list
    """
    with open(file_name_path, 'w') as f:
        for v in vertexs:
            # print(v)
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            if len(face) == 4:
                f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])) 
            if len(face) == 3:
                f.write("f {} {} {}\n".format(face[0], face[1], face[2])) 


def showTriMeshUsingMatlot(TriVs, TriFace, mlab, colormap=colorMaps[2], opacity=1.0):
    """
    用mayavi 绘制三角面
    :param TriVs: 顶点
    :param TriFace: 点序
    :return:
    """
    if isinstance(TriVs, list):
        TriVs = np.array(TriVs, dtype=np.float32)
    if isinstance(TriFace, list):
        TriFace = np.array(TriFace, dtype=np.int32)
    mlab.triangular_mesh(TriVs[:, 0], TriVs[:, 1], TriVs[:, 2], TriFace-1, colormap=colormap, opacity=opacity) # 注意索引值从0开始
    return mlab


def drawPoints(Points, mlab, scale=0.025, color=(np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0])):
    """
    用MayaVi绘制点
    :param Points: 欲绘制的顶点 n * 3
    :param mlab:
    :return: 返回mlab
    """
    if isinstance(Points, list):
        Points = np.array(Points, dtype=np.float32)
    mlab.points3d(Points[:, 0], Points[:, 1], Points[:, 2], scale_factor=scale, color=color)
    for i in range(0, len(Points)):
        mlab.text3d(Points[i, 0], Points[i, 1], Points[i, 2], str(i+1), scale=scale*1.5, color=(0, 0, 0))
    return mlab


def loadFaceMarkers(FacemarkerPath):
    FaceMarker = scio.loadmat(FacemarkerPath)
    FaceMarker = FaceMarker["Marker"]
    return FaceMarker


def normPts(verts, mean, std):
    """
    normalize verts
    :param verts:
    :param mean:
    :param std:
    :return:
    """
    row, col = verts.shape
    T = np.eye(col+1)
    mu = np.mean(verts, axis=0)
    T[0:col, col] = (mean-mu).T
    mean_distance = np.mean(np.sum(np.sqrt((verts-mu)**2), axis=1), axis=0)
    scale = std/mean_distance
    T = scale * T
    T[col, col] = 1
    verts = np.concatenate((verts, np.ones((row, 1))), axis=1)
    verts = np.dot(T, verts.T).T
    verts = verts[:, 0:col]
    return verts


def resSimXform(b, A, B):
    t = b[4:7]
    R = np.zeros((3, 3))
    R = R_axis_angle(R, b[0:3], b[3])
    rot_A = b[7]*R.dot(A) + t[:, np.newaxis]
    result = np.sqrt(np.sum((B-rot_A)**2, axis=0))
    return result


def similarity_fitting(Points_A, Points_B):
    """
    calculate the R t s between PointsA and PointsB
    :param Points_A: n * 3  ndarray
    :param Points_B: n * 3  ndarray
    :return: R t s
    """
    row, col = Points_A.shape
    if row > col:
        Points_A = Points_A.T  # 3 * n
    row, col = Points_B.shape
    if row > col:
        Points_B = Points_B.T  # 3 * n
    cent = np.vstack((np.mean(Points_A, axis=1), np.mean(Points_B, axis=1))).T
    cent_0 = cent[:, 0]
    cent_0 = cent_0[:, np.newaxis]
    cent_1 = cent[:, 1]
    cent_1 = cent_1[:, np.newaxis]
    X = Points_A - cent_0
    Y = Points_B - cent_1
    S = X.dot(np.eye(Points_A.shape[1], Points_A.shape[1])).dot(Y.T)
    U, D, V = np.linalg.svd(S)
    V = V.T
    W = np.eye(V.shape[0], V.shape[0])
    W[-1, -1] = np.linalg.det(V.dot(U.T))
    R = V.dot(W).dot(U.T)
    t = cent_1 - R.dot(cent_0)
    sigma2 = (1.0 / n) * np.multiply(cent_0, cent_0).sum()
    s = 1.0 / sigma2 * np.trace(np.dot(np.diag(D), W))
    #s = 1.0
    b0 = np.zeros((8,))
    if np.isreal(R).all():
        axis, theta = R_to_axis_angle(R)
        b0[0:3] = axis
        b0[3] = theta
        if not np.isreal(b0).all():
            b0 = np.abs(b0)
    else:
        print("R is {}".format(R))
        os.system("pause")
    b0[4:7] = t.T
    b0[7] = s
    b = least_squares(fun=resSimXform, x0=b0, jac='3-point', method='lm', args=(Points_A, Points_B),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)
    r = b.x[0:4]
    t = b.x[4:7]
    s = b.x[7]
    R = R_axis_angle(R, r[0:3], r[3])
    rot_A = s*R.dot(Points_A) + t[:, np.newaxis]
    res = np.sum(np.sqrt(np.sum((Points_B-rot_A)**2, axis=1)))/Points_B.shape[1]
    print("对齐误差是{}".format(res))
    return R, t, s


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """
    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]
    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = atan2(r, t - 1)
    # Normalise the axis.
    axis = axis / r
    # Return the data.
    return axis, theta


def R_axis_angle(matrix, axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """
    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca
    # Depack the axis.
    x, y, z = axis
    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix


def v4_normal(Vert, Face):
    """
    convert 3 vertices representation to 4 vertices with representation of normal
    :param Vert: n * 3 matrix
    :param Face: m * 3 matrix
    :return: T N V F
    """
    f1 = Face[:, 0] - 1
    f2 = Face[:, 1] - 1
    f3 = Face[:, 2] - 1
    e1 = Vert[f2, :] - Vert[f1, :]
    e2 = Vert[f3, :] - Vert[f1, :]
    c = np.cross(e1, e2)
    c_norm = np.sqrt(np.sum(c**2, axis=1))
    c_norm[np.where(c_norm == 0)] = 1
    N = (c.T/c_norm).T
    v4 = Vert[f1, :] + N
    V = np.vstack((Vert, v4))
    F4 = Vert.shape[0] + np.where(Face[:, 2])[0] + 1
    F = np.hstack((Face, F4[:, np.newaxis]))
    T = []
    for i in range(0, F.shape[0]):  # F.shape[0] triangles
        Q = np.transpose(np.vstack((V[F[i, 1]-1, :] - V[F[i, 0]-1, :], V[F[i, 2]-1, :] - V[F[i, 0]-1, :],
                                   V[F[i, 3]-1, :] - V[F[i, 0]-1, :])))
        T.append(Q)
    return T, N, V, F


def build_adjacency(FS):
    """
    build up the adjacency matrix
    :param FS: Triangle indices of source mesh
    :return: 共享第i个三角形三条边的三个三角形的索引 从 0 开始 numpy array
    """
    Adj_idx = np.zeros((FS.shape[0], 3), dtype=np.int32)
    # Adj_idx = FunForParFor_func(Adj_idx, FS)
    # for i in range(0, FS.shape[0]):
    #     for j in range(0, 3):
    #         idx = np.where(np.sum(FS == FS[i, j], axis=1) & np.sum(FS == FS[i, (j+1) % 3], axis=1))[0]
    #         if np.sum(idx != i):
    #             Adj_idx[i, j] = idx[np.where(idx != i)]
    Parallel(n_jobs=4, backend="threading")(delayed(FunForParFor_func)(Adj_idx, FS, i) for i in range(0, FS.shape[0]))
    return Adj_idx


def FunForParFor_func(Adj_idx, FS, i):
    for j in range(0, 3):
        idx = np.where(np.sum(FS == FS[i, j], axis=1) & np.sum(FS == FS[i, (j + 1) % 3], axis=1))[0]
        if np.sum(idx != i):
            Adj_idx[i, j] = int(idx[np.where(idx != i)])


def build_elementary_cell(TS, len):
    """
     The paper has developed a way to derive the equivalent version of the
  equation above in vertex form, as far as we known, T[i] can be represented as
          T[i] = U[i] * inv V[i],

    where U[i] is the surface matrix of deformed triangle i, verbosely
  represented as:
          U[i] = [u2-u1, u3-u1, u4], u4 = sqrt normalized (u2-u1)x(u3-u1)
    where u1, u2, u3 are vertices of this deformed triangle unit, notice that
  U[i] is linearly represented by the vertices of the triangle, so we can
  rewrite the objective function into a squared sum of three linear expressions
          || U[i] * inv V[i] - U[j0] * inv V[j0] ||_F ^2
        + || U[i] * inv V[i] - U[j1] * inv V[j1] ||_F ^2
        + || U[i] * inv V[i] - U[j2] * inv V[j2] ||_F ^2

    Squared Frobenius norm happens to be the squared sum of all elements of the
  matrix, so that we can reshape each matrix in the expression to a column
  vector and evaluate the squared L2 norm of them.
          U * V
      [u2x-u1x, u3x-u1x, u4x]   [v11, v12, v13]
    = [u2y-u1y, u3y-u1y, u4y] * [v21, v22, v23]
      [u2z-u1z, u3z-u1z, u4z]   [v31, v32, v33]
    = [-(v11 + v21 + v31)*u1x + v11*u2x + v21*u3x + v31*u4x,
       -(v12 + v22 + v32)*u1y + v12*u2y + v22*u3y + v32*u4y,
       -(v13 + v23 + v33)*u1z + v13*u2z + v23*u3z + v33*u4z ]
      [......]
      [......]
    reshape to column vector form (inspect the following equation with a wider screen >_< ):
    = [-(v11 + v21 + v31), v11, v21, v31]                                                                               [u1x]
      [-(v12 + v22 + v32), v12, v22, v32]                 0                                                             [u2x]
      [-(v13 + v23 + v33), v13, v23, v33]                                                                               [u3x]
                                                                                                                        [u4x]
                                          [-(v11 + v21 + v31), v11, v21, v31]                                           [u1y]
                      0                   [-(v12 + v22 + v32), v12, v22, v32]                  0                    *   [u2y]
                                          [-(v13 + v23 + v33), v13, v23, v33]                                           [u3y]
                                                                                                                        [u4y]
                                                                              [-(v11 + v21 + v31), v11, v21, v31]       [u1z]
                      0                                   0                   [-(v12 + v22 + v32), v12, v22, v32]       [u2z]
                                                                              [-(v13 + v23 + v33), v13, v23, v33]       [u3z]
                                                                                                                        [u4z]
    However, rather than representing the 9x9 coefficient matrix "as is" in a
  dt_real_type[9][9], we align all meaningful element of the matrix to the
  left side, so that we can shrink this matrix and stuff it into a smaller 9x4
  matrix. That's what these code are all about.
    :param TS:
    :param len:
    :return: E
    """
    E = [0 for i in range(0, len)]
    for i in range(0, len): 
        V = np.linalg.inv(TS[i])
        E[i] = np.hstack((-np.sum(V, axis=0).T[:, np.newaxis], V.T))
    return E


def build_phase1(Adj_idx, E, FS4, VT4, ws, wi, marker):
    """
    non-grid registration phase1
    :param Adj_idx:
    :param E:
    :param FS4:
    :param VT4:
    :param ws:
    :param wi:
    :param marker:
    :return:
    """
    n_adj = Adj_idx.shape[0] * Adj_idx.shape[1]
    len_col = np.max(FS4)
    I1 = np.zeros((9*n_adj*4, 3))
    I2 = np.zeros((9*n_adj*4, 3))
    I3 = np.zeros((9*len(FS4)*4, 3))
    C1 = np.zeros((9*n_adj, 1))
    C2 = wi*np.tile(np.reshape(np.eye(3), [9, 1]), (FS.shape[0], 1))
    for i in range(0, FS4.shape[0]):
        for j in range(0, 3):
            if Adj_idx[i, j]:
                constid = np.zeros((2, 4))
                for k in range(0, 3):
                    if np.sum(marker[:, 0] == FS4[i, k], axis=0):
                        constid[0, k] = (k+1) * np.sum(marker[:, 0] == FS4[i, k])
                    if np.sum(marker[:, 0] == FS4[Adj_idx[i, j], k]):
                        constid[1, k] = (k+1) * np.sum(marker[:, 0] == FS4[Adj_idx[i, j], k])
                U1 = FS4[i, :]
                U2 = FS4[Adj_idx[i, j], :]
                for k in range(0, 3):
                    row = np.tile(np.linspace(0, 2, 3, dtype=np.int32) + i*27 + j*9 + k*3, [4, 1])
                    col1 = np.tile((U1-1)*3 + k, [3, 1]).T
                    val1 = ws*E[i].T
                    if np.sum(constid[0, :]):
                        C1[np.linspace(0, 2, 3, dtype=np.int32) + i * 27 + j*9 + k*3, 0] = C1[np.linspace(0, 2, 3, dtype=np.int32) + i * 27 +
                                                                              j*9 + k*3, 0] - val1[constid[0, :] > 0, :].flatten() * VT4[marker[marker[:, 0] == U1[constid[0, :] > 0], 1]-1, k]
                        val1[constid[0, :] > 0, :] = 0
                    col2 = np.tile((U2-1)*3 + k, [3, 1]).T
                    val2 = -ws * E[Adj_idx[i, j]].T
                    if np.sum(constid[1, :]):
                        C1[np.linspace(0, 2, 3, dtype=np.int32) + i * 27 + j*9 + k*3, 0] = C1[np.linspace(0, 2, 3, dtype=np.int32) + i * 27 +
                                                                              j*9 + k*3, 0] - val2[constid[1, :] > 0, :].flatten() * VT4[marker[marker[:, 0] == U2[constid[1, :] > 0], 1]-1, k]
                        val2[constid[1, :] > 0, :] = 0
                    I1[np.linspace(0, 11, 12, dtype=np.int32) + i*3*3*3*4 + j*3*3*4 + k*3*4, :] = np.hstack((row.flatten('F')[:, np.newaxis], col1.flatten('F')[:, np.newaxis], val1.flatten('F')[:, np.newaxis]))
                    I2[np.linspace(0, 11, 12, dtype=np.int32) + i * 3 * 3 * 3 * 4 + j * 3 * 3 * 4 + k * 3 * 4, :] = np.hstack((row.flatten('F')[:, np.newaxis], col2.flatten('F')[:, np.newaxis], val2.flatten('F')[:, np.newaxis]))
    I1 = I1[I1[:, 0] >= 0, :]
    I2 = I2[I2[:, 0] >= 0, :]
    M1 = sparse.coo_matrix((I1[:, 2], (I1[:, 0], I1[:, 1])), shape=(9*n_adj, 3*len_col))
    M2 = sparse.coo_matrix((I2[:, 2], (I2[:, 0], I2[:, 1])), shape=(9*n_adj, 3*len_col))
    M3 = M1 + M2

    for i in range(0, FS4.shape[0]):
        U1 = FS4[i, :]
        for k in range(0, 3):
            row = np.tile(np.linspace(0, 2, 3, dtype=np.int32) + i*9 + k*3, [4, 1])
            col1 = np.tile((U1-1)*3 + k, [3, 1]).T
            val1 = wi * E[i].T
            I3[np.linspace(0, 11, 12, dtype=np.int32) + i*3*3*4 + k*3*4, :] = np.hstack((row.flatten('F')[:, np.newaxis], col1.flatten('F')[:, np.newaxis], val1.flatten('F')[:, np.newaxis]))
    M4 = sparse.coo_matrix((I3[:, 2], (I3[:, 0], I3[:, 1])), shape=(9*len(FS4), 3*len_col))
    C = np.vstack((C1, C2))
    M = sparse.vstack([M3, M4])
    return M, C


def calc_vertex_norm(F, NF):
    """
    Calculate vertex normal from adjacent face normal
    :param F: n * 3
    :param NF: n * 3
    :return:
    """
    len_F = np.max(F)
    N = np.zeros((len_F, 3))
    for i in range(0, len_F):
        idx = np.where(np.logical_or(np.logical_or((F[:, 0]-1) == i, (F[:, 1]-1) == i), (F[:, 2]-1) == i))[0]
        if idx.shape[0] == 0:
            print(i)
        N[i, :] = np.sum(NF[idx, :], axis=0)/idx.shape[0]
        N[i, :] = N[i, :]/np.sqrt(np.sum(N[i, :]**2))
    return N


def build_phase2(VS, FS, NS, VT, VTN, marker, wc):
    """
    Build pahase 2 sparse matrix M_P2 closest valid point term with of source vertices (nS)
    triangles(mS) target vertices (nT)
    :param VS: deformed source mesh from previous step nS x 3
    :param FS: triangle index of source mesh mS * 3
    :param NS: triangle normals of source mesh mS * 3
    :param VT: target mesh nT * 3
    :param VTN: Vertex normals of source mesh nT * 3
    :param marker: marker constraint
    :param wc: weight value
    :return: M_P2: (3 * nS) x (3 * (nS + mS)) big sparse matrix
    C_P2: (3 * nS) matrix
    """
    VSN = calc_vertex_norm(FS, NS)
    S_size = VS.shape[0]
    valid_pt = np.zeros((S_size, 2))
    C_P2 = np.zeros((3*S_size, 1))
    for j in range(0, S_size):
        if len(np.where(marker[:, 0]-1 == j)[0]) != 0:
            valid_pt[j, :] = np.array([j, marker[marker[:, 0]-1 == j, 1] - 1], dtype=np.int32)
        else:
            valid_pt[j, :] = np.array([j, find_closest_validpt(VS[j, :], VSN[j, :], VT, VTN)], dtype=np.int32)

        C_P2[np.linspace(0, 2, 3, dtype=np.int32) + j*3, 0] = wc * VT[int(valid_pt[j, 1]), :].T
    M_P2 = sparse.coo_matrix((np.tile(wc, [3*S_size, 1])[:, 0], (np.arange(0, 3*S_size), np.arange(0, 3*S_size))), shape=(3*S_size, 3*(VS.shape[0]+FS.shape[0])))
    return M_P2, C_P2


def find_closest_validpt(spt, snormal, vpts, VTN):
    """

    :param spt:
    :param snormal:
    :param vpts:
    :param VTN:
    :return:
    """
    d = np.sum((np.tile(spt, [vpts.shape[0], 1]) - vpts)**2, axis=1)
    ind = np.argsort(d)
    for i in range(0, d.shape[0]):
        if np.arccos(snormal[np.newaxis, :].dot(VTN[ind[i], :][:, np.newaxis])) < np.pi/2:
            valid = ind[i]
            break
    return valid


def non_rigid_registration(VS, FS, VT, FT, ws, wi, wc, marker, file_name):
    tmean = 0
    tstd = np.sqrt(2)
    VS = normPts(VS, tmean, tstd)
    VT = normPts(VT, tmean, tstd)
    R, t, s = similarity_fitting(VT[marker[:, 1] - 1, :], VS[marker[:, 0] - 1, :])
    VT = VT.dot((s * R).T) + t
    if os.path.exists(file_name):
        VSP2 = load_pickle_file(file_name)
        return VSP2, VT
    else:
        TS, NS, VS4, FS4 = v4_normal(VS, FS)
        TT, NT, VT4, FT4 = v4_normal(VT, FT)
        Adj_idx = build_adjacency(FS)
        E = build_elementary_cell(TS, len(FS))
        print("build phase1....")
        M, C = build_phase1(Adj_idx, E, FS4, VT4, ws, wi, marker)
        print("calculate M\C....")
        VSP1 = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
        VSP1 = np.reshape(VSP1[0], (int(VSP1[0].shape[0]/3), 3))
        VSP1 = VSP1[0:VS.shape[0], :]
        VTN = calc_vertex_norm(FT, NT)
        VSP2 = VSP1
        for i in range(0, len(wc)):
            ws = ws + i * wc[i] / 1000
            TS, NS, VS4, FS4 = v4_normal(VSP2, FS)
            E = build_elementary_cell(TS, len(TS))
            M_P1, C_P1 = build_phase1(Adj_idx, E, FS4, VT4, ws, wi, marker)
            M_P2, C_P2 = build_phase2(VSP2, FS, NS, VT, VTN, marker, wc[i])
            M = sparse.vstack([M_P1, M_P2])
            C = np.vstack((C_P1, C_P2))
            VSP2 = sparse.linalg.lsqr(M, C, iter_lim=10000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
            VSP2 = np.reshape(VSP2[0], (int(VSP2[0].shape[0] / 3), 3))
            VSP2 = VSP2[0:VS.shape[0], :]
        mlab.figure("registation result")
        mymlab = showTriMeshUsingMatlot(VT, FT, mlab, colormap=colorMaps[5])
        mymlab = showTriMeshUsingMatlot(VSP2, FS, mlab, colormap=colorMaps[4])
        save_pickle_file(file_name, VSP2)
        return VSP2, VT


def build_correspondence(VS, FS, VT, FT, maxind, thres, FileNameTo_Save):
    """
    build correspondence using the proximity and face normals of source and target meshes
    :param VS: deformed source mesh matched with target nS * 3
    :param FS: Triangle indices of source mesh mS * 3
    :param VT: Target mesh nT * 3
    :param FT: Triangle indices of target mesh mT * 3
    :param maxind: Maximum correspondence
    :param thres: Distance threshold for correspondence
    :param FileNameTo_Save: string for speed up code
    :return: corres mT * # of correspondence for each triangles of target mesh
    """
    if os.path.exists(FileNameTo_Save):
        corres = load_pickle_file(FileNameTo_Save)
        return corres
    else:
        TS, NS, VS4, FS4 = v4_normal(VS, FS)
        TT, NT, VT4, FT4 = v4_normal(VT, FT)
        VS_C = np.zeros((FS.shape[0], 3))
        VT_C = np.zeros((FT.shape[0], 3))
        for i in range(0, FT.shape[0]):
            VT_C[i, :] = np.mean(VT[FT[i, :]-1, :], axis=0)
        for i in range(0, FS.shape[0]):
            VS_C[i, :] = np.mean(VS[FS[i, :]-1, :], axis=0)
        S_tree = KDTree(VS_C)
        T_tree = KDTree(VT_C)
        corres1 = -np.ones((FT.shape[0], maxind))
        corres2 = -np.ones((FT.shape[0], maxind))
        templength = 0
        len_n = 0
        ## for source to target triangle coresspondence
        rowlen = -1
        for i in range(0, FS.shape[0]):
            _, corresind = T_tree.query(VS_C[i, :], k=maxind, distance_upper_bound=thres)
            corresind = corresind[corresind >= 0]
            corresind = corresind[corresind < T_tree.data.shape[0]]
            len_n = corresind.shape[0]
            corresind[np.sum(np.tile(NS[i, :], [NT[corresind, :].shape[0], 1])*NT[corresind, :], axis=1) >= np.pi/2] = -1
            if len(corresind) != 0:
                for j in range(0, len_n):
                    templength = np.max([rowlen, corres2[corresind[j], :][corres2[corresind[j], :] > -1].shape[0]])
                    rowlen = corres2[corresind[j], :][corres2[corresind[j], :] > -1].shape[0]
                    if rowlen == 10:
                        corres2[corresind[j], rowlen - 1] = i
                    else:
                        corres2[corresind[j], rowlen] = i
        corres2 = corres2[:, 0:templength]

        for i in range(0, FT.shape[0]):
            _, corresind = S_tree.query(VT_C[i, :], k=maxind, distance_upper_bound=thres)
            corresind = corresind[corresind >= 0]
            corresind = corresind[corresind < S_tree.data.shape[0]]
            templength = np.max([len_n, corresind.shape[0]])
            len_n = corresind.shape[0]
            corresind[np.sum(np.tile(NT[i, :], [NS[corresind, :].shape[0], 1]) * NS[corresind, :], axis=1) >= np.pi/2] = -1
            corres1[i, 0:len_n] = corresind[0:len_n]
        corres1 = corres1[:, 0:templength]
        tempcorres = np.hstack((corres1, corres2))
        corres = []
        for i in range(0, FT.shape[0]):
            temp = np.unique(tempcorres[i, :])
            temp = temp[temp >= 0]  # here delete -1 term
            corres.append(temp)
        save_pickle_file(FileNameTo_Save, corres)
    return corres


def deformation_transfer(VS, FS, VT, FT, VS2, FS2, corres):
    """
    deformation transfer
    :param VS:
    :param FS:
    :param VT:
    :param FT:
    :param VS2:
    :param FS2:
    :param corres:
    :return:
    """
    lenFS = FS.shape[0]
    lenFT = FT.shape[0]
    SD = [None] * lenFS
    TS, NS, VS4, FS4 = v4_normal(VS, FS)
    TS2, NS2, VS42, FS42 = v4_normal(VS2, FS2)
    TT, NT, VT4, FT4 = v4_normal(VT, FT)
    for i in range(0, lenFS):
        SD[i] = TS2[i].dot(np.linalg.inv(TS[i]))
    E = build_elementary_cell(TT, FT.shape[0])
    n_corres = sum([corres[i].shape[0] for i in range(0, len(corres))])
    n_non_corres = sum([not corres[i] for i in range(0, len(corres)) if len(corres[i]) == 0])
    I = np.zeros((9*(n_corres + n_non_corres)*4, 3))
    C = np.zeros((9*(n_corres + n_non_corres), 1))
    offset = 0
    offset2 = 0
    for i in range(0, lenFT):
        lenCor = corres[i].shape[0]
        Cor = corres[i]
        U = FT4[i, :]
        if lenCor:
            for j in range(0, lenCor):
                for k in range(0, 3):
                    row = np.tile(np.linspace(0, 2, 3, dtype=np.int32) + offset + j*3*3 + k*3, [4, 1])
                    col1 = np.tile((U-1)*3+k, [3, 1]).T
                    val1 = E[i].T
                    I[np.linspace(0, 11, 12, dtype=np.int32) + offset2 + j*3*3*4 + k*3*4, :] = np.hstack((row.flatten('F')[:, np.newaxis], col1.flatten('F')[:, np.newaxis], val1.flatten('F')[:, np.newaxis]))
                C[np.linspace(0, 8, 9, dtype=np.int32) + offset + 9*j, 0] = SD[int(Cor[j])].T.flatten("F")
            offset = offset + 3*3*lenCor
            offset2 = offset2 + 3*3*lenCor*4
        else:
            for k in range(0, 3):
                row = np.tile(np.linspace(0, 2, 3, dtype=np.int32) + offset + k*3, [4, 1])
                col1 = np.tile((U-1)*3+k, [3, 1]).T
                val1 = E[i].T
                I[np.linspace(0, 11, 12, dtype=np.int32) + offset2 + k*3*4] = np.hstack((row.flatten('F')[:, np.newaxis], col1.flatten('F')[:, np.newaxis], val1.flatten('F')[:, np.newaxis]))
            C[np.linspace(0, 8, 9, dtype=np.int32) + offset, 0] = np.eye(3).flatten("F")
            offset = offset + 3*3
            offset2 = offset2 + 3*3*4
    M = sparse.coo_matrix((I[:, 2], (I[:, 0], I[:, 1])), shape=(9*(n_corres + n_non_corres), 3*VT4.shape[0]))
    x = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    x = np.reshape(x[0], (int(x[0].shape[0] / 3), 3))
    x = x[0:VT.shape[0], :]
    temp, nx, v, f = v4_normal(x, FT)
    return x, nx


def demo():
    start = time.time()
    objpath = './face-poses\\face-reference.obj'
    VS, FS = loadObj(objpath)
    objpath = "./face-poses\\face-03-fury.obj"
    VS2, FS2 = loadObj(objpath)
    target_objpath = './head-poses\\head-reference.obj'
    VT, FT = loadObj(target_objpath)
    FacemarkerPath = "./Face_Marker.mat"
    marker = loadFaceMarkers(FacemarkerPath)
    # mymlab = showTriMeshUsingMatlot(s_verts, s_face, mlab)
    # points = verts[FaceMarker[:, 0]-1, :]  # need reduce 1
    # drawPoints(points, mymlab, scale=0.003)
    ws = 1.0
    wi = 5.0  # smooth
    wc = [1, 500, 3000, 5000]
    VS_Reg, VT_Reg = non_rigid_registration(VS, FS, VT, FT, ws, wi, wc, marker, "vsp2.pkl")
    corres = build_correspondence(VS_Reg, FS, VT_Reg, FT, 10, 0.05, "Face_ICIP_corres.pkl")
    x, nx = deformation_transfer(VS, FS, VT, FT, VS2, FS2, corres)
    end = time.time()
    print("takes time {}".format(end - start))
    writeObj("flex_048.obj", x.tolist(), FT.tolist())
    mlab.figure("source mesh")
    mymlab = showTriMeshUsingMatlot(VS, FS, mlab, colormap=colorMaps[4])
    mlab.figure("source deformed mesh")
    mymlab = showTriMeshUsingMatlot(VS2, FS2, mlab, colormap=colorMaps[4])
    mlab.figure("target  mesh")
    mymlab = showTriMeshUsingMatlot(VT, FT, mlab, colormap=colorMaps[5])
    mlab.figure("target deformed mesh")
    mymlab = showTriMeshUsingMatlot(x, FT, mlab, colormap=colorMaps[5])
    mlab.show()


if __name__ == '__main__':

    demo()
