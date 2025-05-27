import itertools
#1.2.1. Các phép xử lý với danh sách

# 1. Ghép 2 danh sách
danhsach1 = [1, 2, 3]
danhsach2 = [5.0, 7.1]
danhsach = danhsach1 + danhsach2
print("Ghép nối danh sách:", danhsach)

# 2. Nhân đôi danh sách
danhsach_gapdoi = danhsach * 2
print("Nhân đôi danh sách:", danhsach_gapdoi)

# 3. Chia nửa danh sách (không phải phép toán /)
nua_danhsach = danhsach[:len(danhsach)//2]
print("Nửa đầu danh sách:", nua_danhsach)

# 4. Ghép 3 danh sách bằng zip
mon_hoc = ["ToanCC", "DSTT", "ToanRR", "LaptrinhCB"]
thu_tu = [2, 3, 4, 1]
diem_so = [10, 9, 8, 7]

anh_xa = list(zip(thu_tu, mon_hoc, diem_so))
print("Ghép 3 danh sách bằng zip:", anh_xa)

# 5. Chuyển zip thành tập hợp
tap_hop = set(anh_xa)
print("Tập hợp từ zip:", tap_hop)

# 6. Tách zip ngược lại (unzip)
lay_TT, lay_monhoc, lay_diem = zip(*anh_xa)
print("Các môn học sau unzip:", lay_monhoc)

# 7. Dùng itertools.chain để nối nhiều range
tap_sinh = list(itertools.chain(range(4), range(5, 10), range(15, 20)))
print("Tập sinh từ nhiều range:", tap_sinh)

# 8. Tạo bộ 3 từ 3 dãy bằng zip
bo_ba = list(zip(range(4), range(7, 12), reversed(range(11))))
print("Bộ ba tạo từ 3 danh sách:", bo_ba)

#2.1. Định nghĩa Symbol và các phép toán hình thức (symbolic)


from sympy import Symbol, symbols, expand # type: ignore

# Tạo Symbol đơn
x = Symbol('x')
y = Symbol('y')

# Biểu thức với biến x chưa gán giá trị
f = x + x + x + 2
print("f = x + x + x + 2 =", f)  

# Tạo symbol có tên cụ thể
a = Symbol('Noi')
b = Symbol('Chim')
expr1 = 3 * b + 7 * a
print("expr1 = 3*b + 7*a =", expr1)

# Truy xuất tên thực sự của symbol
print("Tên của a:", a.name)
print("Tên của b:", b.name)

# Tạo nhiều symbol cùng lúc
a2, b2, c2 = symbols('a b c')
print("Symbols tạo bằng symbols():", a2, b2, c2)

# Tạo biểu thức nhân
s = x * y + y * x
print("s = x*y + y*x =", s) 

# Biểu thức nhân lồng
p1 = x * (x + 2 * x)
print("p1 = x*(x + 2*x) =", p1)  

# Biểu thức chưa triển khai
p2 = (x + 2) * (y + 3)
print("p2 = (x + 2)*(y + 3) =", p2)  

# Biểu thức phức tạp hơn
p3 = (x + 2) * (y + 3) + (x + 2) * (x - 3)
print("p3 chưa expand =", p3)

# Sử dụng expand để triển khai biểu thức
expanded = p3.expand()
print("p3 sau khi expand =", expanded)

# Biểu thức đơn giản hóa về 0
p4 = x * (-x + 2 * x - x)
print("p4 = x*(-x + 2*x - x) =", p4) 

#2.2.2. Đọc thêm: Biểu diễn kết quả 

from sympy import Symbol, pprint, init_printing, factor

# Khởi tạo chế độ in theo thứ tự nghịch
init_printing(order='rev-lex')

# Tạo các biến
x = Symbol('x')
y = Symbol('y')

# Tạo biểu thức x**2 + 2*x*y + y**2
bieuthuc = x**2 + 2*x*y + y**2

# In biểu thức ban đầu với pprint
print("Biểu thức ban đầu:")
pprint(bieuthuc)

# Phân tích biểu thức thành nhân tử
bieuthuc1 = factor(bieuthuc)

# In biểu thức sau khi factor
print("\nBiểu thức sau khi phân tích (factor):")
pprint(bieuthuc1)

# 2.2.3. Thay thế giá trị 
#Thực hành: Sinh viên thực hành 3 tình huống sau:
from sympy import Symbol, simplify, sin, cos

# Khởi tạo biến
x = Symbol('x')
y = Symbol('y')

bieuthuc = x**2 * y + 9

print("Biểu thức ban đầu:")
print("bieuthuc =", bieuthuc)

# Tình huống 1: Thay x=3, y=x
giatri = bieuthuc.subs({x:3, y:x})
print("\nTình huống 1: Thay x=3, y=x")
print("giatri =", giatri) 

# Tình huống 2: Thay x=y, y=3
giatri = bieuthuc.subs({x:y, y:3})
print("\nTình huống 2: Thay x=y, y=3")
print("giatri =", giatri) 

# Tình huống 3: Thay y=x rồi x=3
giatri = bieuthuc.subs({y:x}).subs({x:3})
print("\nTình huống 3: Thay y=x rồi x=3")
print("giatri =", giatri) 
# Sử dụng simplify
print("\n--- Ví dụ sử dụng simplify() ---")
bieuthuc2 = x**2 - x*y + y**2
print("Biểu thức ban đầu:", bieuthuc2)

# Thay thế x = 1 - y
bieuthuc_moi = bieuthuc2.subs({x:1 - y})
print("Sau khi thay x = 1 - y:", bieuthuc_moi)

# Đơn giản hóa biểu thức
dongian = simplify(bieuthuc_moi)
print("Biểu thức sau khi simplify:", dongian)

# Ví dụ với biểu thức lượng giác
print("\n--- Ví dụ biểu thức lượng giác ---")
bt = sin(x)*cos(y) + cos(x)*sin(y)
print("Biểu thức ban đầu:", bt)

bt_moi = simplify(bt)
print("Biểu thức sau khi simplify:", bt_moi)  

#3.1. Một số lệnh cơ bản numpy xử lý vector 
import numpy as np

# Tạo vec1
vec1 = np.array([1, 3, 5.])
print("vec1 =", vec1)

# vec1 * 2
print("vec1 * 2 =", vec1 * 2)

# vec1 * vec1
print("vec1 * vec1 =", vec1 * vec1)  # phép nhân từng phần tử (nhân theo từng chiều)

# vec1 / 2
print("vec1 / 2 =", vec1 / 2)

# vec1 + vec1
print("vec1 + vec1 =", vec1 + vec1)

# Tạo vec2
vec2 = np.array([2., 4.])
# vec1 + vec2 (Lỗi do không cùng kích thước)
try:
    print("vec1 + vec2 =", vec1 + vec2)
except ValueError as e:
    print("vec1 + vec2 gây lỗi do không cùng kích thước:", e)

# Tạo vec3
vec3 = np.array([2., 4., 6.])
print("vec1 + vec3 =", vec1 + vec3)

print("vec1 / vec3 =", vec1 / vec3)

print("vec1 * vec3 =", vec1 * vec3)

print("2 * vec1 + 5 * vec3 =", 2 * vec1 + 5 * vec3)

# Truy xuất phần tử
print("vec3[2] =", vec3[2])

# Tạo vector bằng linspace
vec4 = np.linspace(0, 20, 5)
print("vec4 =", vec4)

# Vector toàn 0
vec5 = np.zeros(5)
print("vec5 =", vec5)

# Vector toàn 1
vec6 = np.ones(5)
print("vec6 =", vec6)

# Vector rỗng (giá trị ngẫu nhiên chưa khởi tạo)
vec7 = np.empty(5)
print("vec7 =", vec7)

# Vector ngẫu nhiên từ 0 đến 1
vec8 = np.random.random(5)
print("vec8 =", vec8)

# Xử lý vector
v = np.array([1., 3., 5.1])
print("Tổng các phần tử của v =", np.sum(v))
print("Số chiều của v =", v.shape)

# Thử chuyển vị vector
print("v.transpose() =", v.transpose())  # không thay đổi vì vector 1 chiều

# Lấy phần của vector
v1 = v[:2]
print("v1 =", v1)

# Gán lại phần tử
v[0] = 5
print("v sau khi v[0] = 5:", v)
print("v1 sau khi gán:", v1)

# Thử gán sai kiểu (lỗi)
try:
    v1[2] = [1, 2, 3]
except Exception as e:
    print("Lỗi khi gán v1[2] =", e)

# Phép cộng số và vector
print("v + 10.0 =", v + 10.0)

# Căn bậc 2
print("np.sqrt(v) =", np.sqrt(v))

# Hàm cos
print("np.cos(v) =", np.cos(v))

# Hàm sin
print("np.sin(v) =", np.sin(v))

# Tích vô hướng
v1 = np.array([1., 4.6])
v3 = np.array([2., 5.])
print("np.dot(v1, v3) =", np.dot(v1, v3))
print("v1.dot(v3) =", v1.dot(v3))
print("v3.dot(v1) =", v3.dot(v1))

#################### BÀI TẬP CHƯƠNG 1#######################
# Bài tập 1: Giải phương trình với Sympy 
from sympy import Symbol, solve

x = Symbol('x')

bieuthuc = x + 3 - 5
print(solve(bieuthuc))

bieuthuc = x**2 + 3 - 5
nghiemx = solve(bieuthuc)
print(nghiemx)
print(nghiemx[0])
print(nghiemx[1])

# Bài tập 2: Giải phương trình bậc 2 
x = Symbol('x')

# Phương trình bậc 2: x^2 + 9x + 8 = 0

ptb2 = x**2 + 9*x + 8
nghiem1 = solve(ptb2, dict=True)
print(nghiem1)

# Phương trình bậc 2 có nghiệm ảo: x^2 + x + 10 = 0
ptb2_ao = x**2 + 1*x + 10
nghiem2 = solve(ptb2_ao, dict=True)
print(nghiem2)

# Bài tập 3: Giải phương trình 1 biến biểu diễn đại số các biến còn lại 

x = Symbol('x')
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')

ptb2 = a*x**2 + b*x + c

nghiem = solve(ptb2, x, dict=True)
print(nghiem)

# Bài tập 4: Giải hệ phương trình

x = Symbol('x')
y = Symbol('y')

pt1 = 2*x + 3*y - 12
pt2 = 3*x - 2*y - 5

nghiem = solve((pt1, pt2), dict=True)
print(nghiem)

nghiem = nghiem[0]

# Kiểm tra nghiệm bằng cách thay vào từng phương trình
kq_pt1 = pt1.subs({x: nghiem[x], y: nghiem[y]})
kq_pt2 = pt2.subs({x: nghiem[x], y: nghiem[y]})

print(kq_pt1)
print(kq_pt2)

# Bài tập 5: Thể hiện ma trận bằng numpy

import numpy as np  

# Khai báo ma trận M1
M1 = np.array([[9, 12], [23, 30]])
print("M1 =")
print(M1)
print()

# Khai báo vector u
u = np.array([2, 1])

# Tích ma trận M1 với vector u
tichM1u = M1.dot(u)
print("M1.dot(u) =")
print(tichM1u)
print("Giải thích: Tích ma trận M1 (2x2) với vector cột u (2x1) ra vector 2x1.")
print()

# Tích vector u với ma trận M1
tichuM1 = u.dot(M1)
print("u.dot(M1) =")
print(tichuM1)
print("Giải thích: Tích vector hàng u (1x2) với ma trận M1 (2x2) ra vector 1x2.")
print()

# Dùng hàm np.dot cho hai trường hợp trên
print("np.dot(M1, u) =")
print(np.dot(M1, u))
print("Giải thích: np.dot(M1,u) giống M1.dot(u), tích ma trận M1 với vector cột u.")
print()

print("np.dot(u, M1) =")
print(np.dot(u, M1))
print("Giải thích: np.dot(u,M1) giống u.dot(M1), tích vector hàng u với ma trận M1.")
print()

# Các lệnh numpy khác
mat1 = np.zeros([5,5])
print("mat1 = np.zeros([5,5]) =")
print(mat1)
print("mat1 là ma trận 5x5 toàn 0.")
print()

mat2 = np.ones([5,5])
print("mat2 = np.ones([5,5]) =")
print(mat2)
print("mat2 là ma trận 5x5 toàn 1.")
print()

mat3 = mat1 + 2*mat2
print("mat3 = mat1 + 2*mat2 =")
print(mat3)
print("mat3 là ma trận 5x5 toàn giá trị 2 (0 + 2*1).")
print()

mat4 = mat3
mat3[3][2] = 10
print("Sau mat3[3][2] = 10:")
print("mat3 =")
print(mat3)
print("mat4 =")
print(mat4)
print("Giải thích: mat4 là tham chiếu đến mat3, nên mat4 thay đổi khi mat3 thay đổi.")
print()

mat5 = np.copy(mat3)
mat3[3][2] = 20
print("Sau mat3[3][2] = 20:")
print("mat3 =")
print(mat3)
print("mat4 =")
print(mat4)
print("mat5 =")
print(mat5)
print("Giải thích: mat5 là bản sao độc lập, nên không thay đổi khi mat3 thay đổi.")
print()

mat6 = np.empty([4,5])
print("mat6 = np.empty([4,5]) =")
print(mat6)
print("mat6 chứa giá trị rác trong bộ nhớ, chưa được khởi tạo cụ thể.")
print()

mat7 = np.identity(4)
print("mat7 = np.identity(4) =")
print(mat7)
print("mat7 là ma trận đơn vị 4x4 với đường chéo chính là 1, các phần tử còn lại là 0.")
print()

try:
    mat8 = np.rand([4,5])
except AttributeError:
    mat8 = np.random.random([4,5])
print("mat8 = np.random.random([4,5]) =")
print(mat8)
print("mat8 là ma trận 4x5 chứa các số thực ngẫu nhiên trong khoảng [0,1).")
# Bài tập 6:



# Dữ liệu bản đồ nguy cơ (4x5)
A = np.array([[1, 1, 0, 0, 1],
              [3, 1, 0, 1, 1], 
              [5, 2, 0, 1, 2],
              [2, 0, 1, 2, 3]])

B = np.array([[1, 1, 2, 2, 1],
              [2, 2, 2, 0, 2],
              [0, 1, 2, 4, 2],
              [1, 4, 1, 2, 2]])

C = np.array([[0, 5, 1, 1, 1],
              [0, 1, 1, 1, 3],
              [1, 3, 1, 3, 1],
              [0, 1, 3, 3, 0]])

D_map = np.array([[3, 1, 1, 0, 1],
                  [5, 0, 0, 3, 7],
                  [7, 0, 0, 3, 5],
                  [5, 0, 3, 5, 3]])

E = np.array([[0, 0, 0, 10, 0],
              [0, 0, 15, 0, 0],
              [0, 5, 15, 5, 0],
              [0, 20, 5, 0, 0]])


score_lookup = {
    'A': [0, 1, 2, 3, 5],    # Cháy rừng
    'B': [0, 1, 2, 4, 8],    # Lũ quét
    'C': [0, 1, 3, 5, 9],    # Sạt lở núi
    'D': [0, 1, 3, 5, 7],    # Bệnh dịch
    # E là điểm thật
}

def calculate_risk_matrix(A, B, C, D, E):
    rows, cols = A.shape
    total_risk = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            a, b, c, d, e = A[i, j], B[i, j], C[i, j], D[i, j], E[i, j]

            risk = 0
            risk += score_lookup['A'][a] if 0 <= a < 5 else 0
            risk += score_lookup['B'][b] if 0 <= b < 5 else 0
            risk += score_lookup['C'][c] if 0 <= c < 5 else 0
            risk += score_lookup['D'][d] if 0 <= d < 5 else 0
            risk += e  # E là điểm thật

            total_risk[i, j] = risk

    return total_risk

def find_safe_positions(risk_matrix, threshold=5):
    safe = []
    rows, cols = risk_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if risk_matrix[i, j] <= threshold:
                safe.append((i, j))
    return safe

# Tính toán
risk = calculate_risk_matrix(A, B, C, D_map, E)
safe_cells = find_safe_positions(risk)

# Kết quả
print("Tổng điểm nguy cơ:\n", risk)
print("\nCác vị trí an toàn (≤ 5):", safe_cells)
