import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X1, y1 = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

lin_reg1 = LinearRegression()
lin_reg1.fit(X1, y1)
y_pred_lin1 = lin_reg1.predict(X1)

poly = PolynomialFeatures(degree=2, include_bias=False)
X1_poly = poly.fit_transform(X1)

poly_reg1 = LinearRegression()
poly_reg1.fit(X1_poly, y1)

sort_idx = X1.ravel().argsort()
X1_sorted = X1[sort_idx]
y_pred_lin1_sorted = y_pred_lin1[sort_idx]
y_pred_poly1_sorted = poly_reg1.predict(X1_poly[sort_idx])

plt.figure(figsize=(10, 6))
plt.scatter(X1, y1, color="black", label="Вихідні дані", alpha=0.6)
plt.plot(
    X1_sorted, y_pred_lin1_sorted, color="blue", label="Лінійна регресія", lw=2
)
plt.plot(
    X1_sorted,
    y_pred_poly1_sorted,
    color="red",
    label="Поліноміальна регресія (ступінь 2)",
    lw=2,
)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1: Порівняння лінійної та поліноміальної регресії")
plt.show()

X2, y2 = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

degrees = [2, 5, 10]
results2 = []

for degree in degrees:
  poly_feat = PolynomialFeatures(degree=degree, include_bias=False)
  X2_train_poly = poly_feat.fit_transform(X2_train)
  X2_test_poly = poly_feat.transform(X2_test)

  model2 = LinearRegression()
  model2.fit(X2_train_poly, y2_train)

  r2_train = r2_score(y2_train, model2.predict(X2_train_poly))
  r2_test = r2_score(y2_test, model2.predict(X2_test_poly))
  results2.append({"Degree": degree, "R2_train": r2_train, "R2_test": r2_test})

print("Результати Завдання 2:")
print(f"{'Ступінь':<10} | {'R2 (навчальна)':<18} | {'R2 (тестова)':<15}")
print("-" * 50)
for res in results2:
  print(
      f"{res['Degree']:<10} | {res['R2_train']:<18.4f} |"
      f" {res['R2_test']:<15.4f}"
  )
print("\n" + "=" * 50 + "\n")

X3, y3 = make_regression(
    n_samples=500, n_features=20, n_informative=5, noise=10, random_state=42
)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

lin_reg3 = LinearRegression()
lin_reg3.fit(X3_train, y3_train)

lasso_reg3 = Lasso(alpha=0.1, random_state=42)
lasso_reg3.fit(X3_train, y3_train)

r2_lin3 = r2_score(y3_test, lin_reg3.predict(X3_test))
r2_lasso3 = r2_score(y3_test, lasso_reg3.predict(X3_test))

zero_count3 = np.sum(lasso_reg3.coef_ == 0)

print("Результати Завдання 3:")
print(f"R^2 (Звичайна лінійна регресія): {r2_lin3:.4f}")
print(f"R^2 (Lasso регресія): {r2_lasso3:.4f}")
print(f"Кількість занулених коефіцієнтів Lasso: {zero_count3}")