from flint import arb, ctx


class DualArb:
    """
    Dual number over Arb:
        x + eps * dx  (eps^2 = 0)

    val: arb  — 実数部 x
    der: arb  — 微分値 dx
    """

    __slots__ = ("val", "der")

    def __init__(self, val=0, der=0):
        self.val = arb(val)
        self.der = arb(der)

    # --- ユーティリティ ---

    @staticmethod
    def lift(x):
        """スカラー or DualArb を DualArb に持ち上げる。"""
        if isinstance(x, DualArb):
            return x
        return DualArb(x, 0)

    def copy(self):
        return DualArb(self.val, self.der)

    def __repr__(self):
        return f"DualArb(val={self.val}, der={self.der})"

    # --- 四則演算 ---

    def __add__(self, other):
        other = DualArb.lift(other)
        return DualArb(self.val + other.val, self.der + other.der)

    __radd__ = __add__

    def __sub__(self, other):
        other = DualArb.lift(other)
        return DualArb(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        other = DualArb.lift(other)
        return DualArb(other.val - self.val, other.der - self.der)

    def __neg__(self):
        return DualArb(-self.val, -self.der)

    def __mul__(self, other):
        other = DualArb.lift(other)
        # (x + eps dx)(y + eps dy) = xy + eps (x dy + y dx)
        val = self.val * other.val
        der = self.val * other.der + self.der * other.val
        return DualArb(val, der)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = DualArb.lift(other)
        # (x + eps dx) / (y + eps dy)
        # = (x / y) + eps * ( (dx*y - x*dy) / y^2 )
        inv = 1 / other.val
        val = self.val * inv
        der = (self.der * other.val - self.val * other.der) * (inv * inv)
        return DualArb(val, der)

    def __rtruediv__(self, other):
        # (a) / (x + eps dx)
        other = DualArb.lift(other)
        inv = 1 / self.val
        val = other.val * inv
        der = (other.der * self.val - other.val * self.der) * (inv * inv)
        return DualArb(val, der)

    # --- 累乗 ---

    def __pow__(self, n):
        """
        (x + eps dx)^n, n はスカラー（int/float/arb）だけ対応。
        d/dx x^n = n x^(n-1) を使う。
        """
        if isinstance(n, DualArb):
            raise TypeError("DualArb ** DualArb は未実装（スカラー指数だけ対応）。")

        n_arb = arb(n)
        val = self.val ** n_arb
        der = n_arb * (self.val ** (n_arb - 1)) * self.der
        return DualArb(val, der)

    def __rpow__(self, base):
        """
        base ** (x + eps dx)
        = exp((x + eps dx) * log(base))
        """
        base_arb = arb(base)
        if base_arb <= 0:
            raise ValueError("base は正の実数にしてね（log が必要になるので）。")

        # f(x) = base^x, f'(x) = base^x * log(base)
        val = base_arb ** self.val
        der = val * self.der * base_arb.log()
        return DualArb(val, der)

# --- elementary functions on DualArb ---

def dual_exp(x):
    """
    exp(x + eps dx) = exp(x) + eps * dx * exp(x)
    """
    x = DualArb.lift(x)
    v = x.val.exp()
    return DualArb(v, x.der * v)

def dual_log(x):
    """
    log(x + eps dx) = log(x) + eps * dx / x
    """
    x = DualArb.lift(x)
    return DualArb(x.val.log(), x.der / x.val)

def dual_sin(x):
    """
    sin(x + eps dx) = sin(x) + eps * dx * cos(x)
    """
    x = DualArb.lift(x)
    return DualArb(x.val.sin(), x.der * x.val.cos())

def dual_cos(x):
    """
    cos(x + eps dx) = cos(x) - eps * dx * sin(x)
    """
    x = DualArb.lift(x)
    return DualArb(x.val.cos(), -x.der * x.val.sin())

def dual_tan(x):
    """
    tan = sin / cos で実装（安直版）
    """
    return dual_sin(x) / dual_cos(x)

def dual_sqrt(x):
    """
    sqrt(x + eps dx) = sqrt(x) + eps * dx / (2 * sqrt(x))
    """
    x = DualArb.lift(x)
    v = x.val.sqrt()
    return DualArb(v, x.der / (2 * v))



def differentiate(f, x0, dps=50):
    ctx.dps = dps
    x = DualArb(x0, 1)
    y = f(x)
    if not isinstance(y, DualArb):
        y = DualArb.lift(y)
    return y.val, y.der

if __name__ == "__main__":
    from flint import arb

    # f(x) を DualArb で書く
    def f(x: DualArb) -> DualArb:
        # f(x) = x^3 + 2 * sin(x) * exp(x)
        return x**3 + 2 * dual_sin(x) * dual_exp(x)

    # xの値と精度を入力
    val, der = differentiate(f, x0=2, dps=80)

    # 結果出力
    print("-------------- 自動微分の結果出力 -------------------")
    print("f(x)  =", val)
    print("f'(x) =", der)


    # ここからは検算用（普通のarbで計算）

    # 検算用の x の 値をarbで入力
    x = arb(2)
    f_exact = x**3 + 2 * x.sin() * x.exp()
    # f'(x) = 3x^2 + 2[cos x * e^x + sin x * e^x]
    fprime_exact = 3 * x**2 + 2 * (x.cos() * x.exp() + x.sin() * x.exp())

    # 検算出力
    print("-------------- 検算用の結果出力 ---------------------")
    print("f(x) (exact-like)  =", f_exact)
    print("f'(x) (exact-like) =", fprime_exact)
