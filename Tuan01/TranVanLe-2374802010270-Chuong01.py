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


from sympy import Symbol, symbols, expand

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