import os
import mysql.connector
from dotenv import load_dotenv
import csv


# IDEA use the base model to figure out roughly the ratio of utterances with orgs vs utterances without orgs

def createConnection():
    load_dotenv()
    print(os.getenv("mydb"))
    connection = mysql.connector.connect(
        host=os.getenv("myhost"),
        user=os.getenv("myuser"),
        password=os.getenv("mypass"),
        database=os.getenv("mydb"),
        connection_timeout=300,  # Set the connection timeout in seconds
        buffered=True            # Ensures complete result sets are fetched
    )
    return connection


def main():
    connection = createConnection()

    cursor = connection.cursor()
    # Prepare CSV file
    with open('allOrgs.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        # Write header
        csvwriter.writerow(['oid', 'name'])
        cursor.execute(
            f"SELECT oid,name from Organizations")
        rows_table = cursor.fetchall()  # Fetch the results of the first query
        # Write rows to the CSV
        for row in rows_table:
            print(f"id: {row[0]} {row[1]}")
            # csvwriter.writerow(row)

    cursor.close()
    connection.close()


# Call the main function
if __name__ == "__main__":
    main()
