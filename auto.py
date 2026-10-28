import numpy as np

rng = np.random.default_rng(42)

price = np.arange(10,63,1).reshape(-1,1)

#Робимо залежність між ціною авто і продажем
base_sales = 480*np.exp(-0.28*price.ravel())
print("price: ", len(price))
print("price: ", price)

print("base_sales: ", len(base_sales))

#Додаємо випадковий шум
noise = rng.normal(0,6,size=price.shape[0])
print("noise = ",noise)
sales = np.round(base_sales + noise).astype(int)
print(sales)
sales = np.clip(sales,1, None)
print(sales)