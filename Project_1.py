import pandas as pd # STEP No.1
import matplotlib.pyplot as plt # STEP No.2 & 3
from sklearn.model_selection import train_test_split # STEP No.3
from sklearn.linear_model import LinearRegression # STEP No.3
from sklearn.metrics import mean_squared_error, r2_score # STEP No.4


# STEP No.1: Dataset
data = pd.read_csv('tips.csv') #load

print(data.head()) #Print first five Rows
print(data.shape) #Print Shape (Row,Column)
data.info()

print(data.value_counts())
print(data.isnull().sum())
print(data['total_bill'].value_counts())
print(data['tip'].value_counts())
print(data['sex'].value_counts())
print(data['smoker'].value_counts())
print(data['day'].value_counts())
print(data['time'].value_counts())
print(data['size'].value_counts())

# STEP No.2: Data Visulaization

x = data[['total_bill']]
y = data['tip']
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
axs[0].scatter(x, y)
axs[0].set_xlabel('Total Bill')
axs[0].set_ylabel('Tip')
axs[0].set_title('Data')

# STEP No.3: Model Building

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=44)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#   Train Data
axs[1].scatter(x_train, y_train)
axs[1].plot(x_train,model.predict(x_train), color = 'purple')
axs[1].set_xlabel('Total Bill')
axs[1].set_ylabel('Tip')
axs[1].set_title('Train Data')

#   Test Data
axs[2].scatter(x_test, y_test)
axs[2].plot(x_test,model.predict(x_test), color = 'Black')
axs[2].set_xlabel('Total Bill')
axs[2].set_ylabel('Tip')
axs[2].set_title('Test Data')

plt.savefig('plot.png', bbox_inches='tight')
plt.show()

# STEP No.4: Model Evaluation

mse = mean_squared_error(y_test, y_pred) # Mean Squared Error
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred) # R-Squared
print(f'R-squared: {r2}')

# Compare the Actual with the Predicted Values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())