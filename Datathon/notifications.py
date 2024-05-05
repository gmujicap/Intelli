import csv
from collections import defaultdict
from datetime import datetime
from calendar import monthrange

def calculate_averages(days, months):
    total_days = len(days)
    total_months = len(months)
    monthDailyAverages = defaultdict(int)

    # Calculate average entries per day
    total_entries_per_day = sum(days.values())
    dayAverage = total_entries_per_day / total_days if total_days > 0 else 0

    # Calculate average entries per month
    total_entries_per_month = sum(months.values())
    monthAverage = total_entries_per_month / total_months if total_months > 0 else 0

    # Calculate average entries per day for each month
    for month_key, entry_count in months.items():
        year, month = month_key.split('-')
        days_in_month = monthrange(int(year), int(month))[1]  # Get number of days in the month
        total_entries_in_month = sum(days.get(day, 0) for day in days if day.startswith(month_key))
        average_entries_per_day = total_entries_in_month / days_in_month if days_in_month > 0 else 0
        monthDailyAverages[month_key] = average_entries_per_day
    return dayAverage, monthAverage, monthDailyAverages

def read_csv_file(file_path):
    #se crean diccionarios que ligan datos a fechas especificas
    entries_total = defaultdict(int)
    entries_per_day = defaultdict(int)
    entries_per_month = defaultdict(int)
    complaint_per_day = defaultdict(int)

    #cuenta de línea leída para fines de checar errores
    line_number = 0

    #se lee la informacion del csv que incluye los comentarios
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            line_number += 1  # Incrementa line number

            # Se categoriza la entrada por tipo
            if row[4] == 'halago':
                entries_total['halago'] += 1
            elif row[4] == 'queja':
                entries_total['queja'] += 1
                complaint_per_day[row[0]] += 1
            elif row[4] == 'soporte':
                entries_total['soporte'] += 1
            elif row[4] == 'producto':
                entries_total['producto'] += 1

            # Se categoriza la entrada por fecha, primero dia luego mes
            try: 
                date = datetime.strptime(str(row[0]), '%d/%m/%Y')  # Parse string to datetime object

                # Count entries per day
                day_key = date.date().isoformat()  # Using ISO format (YYYY-MM-DD) as key
                entries_per_day[day_key] += 1

                # Count entries per month
                month_key = date.strftime('%Y-%m')  # Using YYYY-MM format as key
                entries_per_month[month_key] += 1
            except ValueError as e:
                try:
                    # Try parsing in different format
                    date = datetime.strptime(row[0], '%-d/%-m/%Y')  # Parse string to datetime object
                    day_key = date.date().isoformat()  # Using ISO format (YYYY-MM-DD) as key
                    entries_per_day[day_key] += 1
                    month_key = date.strftime('%Y-%m')  # Using YYYY-MM format as key
                    entries_per_month[month_key] += 1
                except ValueError:
                    print(f"Error parsing date on line {line_number}: {e}")  # Print error message with line number

    data = [entries_total, entries_per_day, entries_per_month, complaint_per_day]
    return data

# Dias y meses con cantidad elevada de tweets
def obtain_anomalies1(csv_data, averages_results):
    abnormalDays = []
    abnormalMonths = []
    errorType = 1

    for day, entry_count in csv_data[1].items():
        month_key = day[:7]  # Extract month key (YYYY-MM) from day
        average_entries_per_day = averages_results[2].get(month_key, 0)
        if entry_count > average_entries_per_day * 1.5:
            abnormalDays.append((day, entry_count, errorType))
    for month, entry in csv_data[2].items():
        if entry > averages_results[1] * 1.5:
            abnormalMonths.append((month, entry, errorType))
    return abnormalDays, abnormalMonths

# Meses con un cambio drástico en interacciones
def obtain_anomalies2():
    print("lol")

# Días con cantidad elevadas de quejas
def obtain_anomalies3(dataset):
    abnormalDays = []
    errorType = 3

    for date, amount in dataset.items():
        if amount > 5:
            abnormalDays.append((date, amount, errorType))
    return abnormalDays

def write_list_to_csv(data_list, output_file):
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(('Date', 'Tweets', 'Type'))
        for item in data_list:
            csv_writer.writerow(item)

file_path = 'dataset_actualizado.csv'
csv_data = read_csv_file(file_path)
#print("A continuacion se muestran todos los datos leídos del CSV, mostrando las fechas con su cantidad de entradas respectivas")
#print(csv_data)

print("\nAhora, se presentan los promedios de entradas por dia y por mes:")
averages_results = calculate_averages(csv_data[1], csv_data[2])
print("\nPromedio de entradas por dia: " + str(averages_results[0]))
print("Promedio de entradas por mes: " + str(averages_results[1]))

# Anomalias tipo 1
print("\nPor ultimo, se presentan los dias con anomalias: ")
anomalias1 = obtain_anomalies1(csv_data, averages_results)
print("\nDias que presentan anomalias Tipo 1: ")
print(anomalias1[0])
print("\nMeses que presentan anomalias")
print(anomalias1[1])

# Anomalias tipo 3
print("\nAnomalias tipo 3: ")
anomalias3 = obtain_anomalies3(csv_data[3])
print("Dias que presentan anomalias Tipo 3: ")
print(anomalias3)

print("Hecho esto, se creara un csv con la informacion de las anomalias: ")
data_list = anomalias1[0] + anomalias3
output_file = 'anomalias_dias.csv'
write_list_to_csv(data_list, output_file)


data_list = anomalias1[1]
output_file = 'anomalias_meses.csv'
write_list_to_csv(data_list, output_file)

