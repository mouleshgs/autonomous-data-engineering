import pandas as pd
import numpy as np
from pathlib import Path

# Target directory
output_dir = Path("testing-db")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. DATASET 1: E-Commerce Retail Orders (CSV)
# -------------------------------------------------------------
csv_records = [
    {
        "Order ID #": "ORD-1001",
        "Customer Full Name": "  johnathan DOE ",
        "Order Date": "2024/01/15",
        "Region Code": "us",
        "Product Category": "  electronics ",
        "Item Quantity": 2,
        "Unit Price ($)": "$249.99",
        "Is Return?": "no",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1002",
        "Customer Full Name": "SARAH CONNOR",
        "Order Date": "2024-02-18",
        "Region Code": " US ",
        "Product Category": "HOME & APPLIANCES",
        "Item Quantity": 1,
        "Unit Price ($)": "$89.50",
        "Is Return?": "YES",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1003",
        "Customer Full Name": "rajesh  kumar",
        "Order Date": "14-03-2024",
        "Region Code": "ind",
        "Product Category": "apparel & fashion",
        "Item Quantity": -3,  # Invalid negative quantity
        "Unit Price ($)": "$45.00",
        "Is Return?": "false",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1004",
        "Customer Full Name": "EMMA WATSON",
        "Order Date": "2024/04/05",
        "Region Code": " uk ",
        "Product Category": "  books & stationery ",
        "Item Quantity": 4,
        "Unit Price ($)": None,  # Missing price
        "Is Return?": "0",
        "Internal Notes (Empty)": None,
    },
    {
        # Intentional duplicate of ORD-1005 below
        "Order ID #": "ORD-1005",
        "Customer Full Name": "Michael Scott",
        "Order Date": "2024-05-12",
        "Region Code": "ca",
        "Product Category": "Office Supplies",
        "Item Quantity": 10,
        "Unit Price ($)": "$12.75",
        "Is Return?": "1",
        "Internal Notes (Empty)": None,
    },
    {
        # Intentional exact duplicate
        "Order ID #": "ORD-1005",
        "Customer Full Name": "Michael Scott",
        "Order Date": "2024-05-12",
        "Region Code": "ca",
        "Product Category": "Office Supplies",
        "Item Quantity": 10,
        "Unit Price ($)": "$12.75",
        "Is Return?": "1",
        "Internal Notes (Empty)": None,
    },
    {
        # Fully empty row (all None)
        "Order ID #": None,
        "Customer Full Name": None,
        "Order Date": None,
        "Region Code": None,
        "Product Category": None,
        "Item Quantity": None,
        "Unit Price ($)": None,
        "Is Return?": None,
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1006",
        "Customer Full Name": "fiona gallagher",
        "Order Date": "06/20/2024",
        "Region Code": "us",
        "Product Category": "beauty & health",
        "Item Quantity": None,  # Missing quantity
        "Unit Price ($)": "$35.00",
        "Is Return?": "No",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1007",
        "Customer Full Name": "  LUCAS MOURA ",
        "Order Date": "2024-07-04",
        "Region Code": None,  # Missing region
        "Product Category": None,  # Missing category
        "Item Quantity": 1,
        "Unit Price ($)": "$199.00",
        "Is Return?": "TRUE",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1008",
        "Customer Full Name": "priya sharma",
        "Order Date": "18-08-2024",
        "Region Code": "IND",
        "Product Category": "Electronics",
        "Item Quantity": -1,  # Negative quantity
        "Unit Price ($)": "$520.00",
        "Is Return?": "y",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1009",
        "Customer Full Name": "David Attenborough",
        "Order Date": "2024/09/25",
        "Region Code": "uk",
        "Product Category": "  outdoors & sports ",
        "Item Quantity": 3,
        "Unit Price ($)": "$145.50",
        "Is Return?": "n",
        "Internal Notes (Empty)": None,
    },
    {
        "Order ID #": "ORD-1010",
        "Customer Full Name": "elena rostova",
        "Order Date": "2024-10-31",
        "Region Code": "ca",
        "Product Category": "home & appliances",
        "Item Quantity": 2,
        "Unit Price ($)": "$75.20",
        "Is Return?": "yes",
        "Internal Notes (Empty)": None,
    },
]

df_csv = pd.DataFrame(csv_records)
csv_path = output_dir / "customer_orders_dirty.csv"
df_csv.to_csv(csv_path, index=False)
print(f"Created CSV: {csv_path.resolve()}")


# -------------------------------------------------------------
# 2. DATASET 2: Hospital Inpatient Admissions & Billing (Excel .xlsx)
# -------------------------------------------------------------
excel_records = [
    {
        "Patient Record #": "MED-8801",
        "Patient Full Name": "  eleanor VANCE ",
        "Admission Date": "2024-01-10",
        "State Code": "ny",
        "Department Name": "  cardiology ",
        "Patient Age": 58,
        "Total Bill Amount ($)": "$4,250.00",
        "Insurance Covered?": "yes",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8802",
        "Patient Full Name": "MARCUS BRODY",
        "Admission Date": "15/02/2024",
        "State Code": " CA ",
        "Department Name": "EMERGENCY MEDICINE",
        "Patient Age": 34,
        "Total Bill Amount ($)": "$1,890.50",
        "Insurance Covered?": "TRUE",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8803",
        "Patient Full Name": "amara kanu",
        "Admission Date": "2024/03/22",
        "State Code": "tx",
        "Department Name": "  pediatrics ",
        "Patient Age": -4,  # Invalid negative age
        "Total Bill Amount ($)": "$920.00",
        "Insurance Covered?": "no",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8804",
        "Patient Full Name": "GREGORY HOUSE",
        "Admission Date": "2024-04-18",
        "State Code": " fl ",
        "Department Name": "diagnostic medicine",
        "Patient Age": 52,
        "Total Bill Amount ($)": None,  # Missing billing amount
        "Insurance Covered?": "1",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        # Intentional duplicate of MED-8805 below
        "Patient Record #": "MED-8805",
        "Patient Full Name": "clara oswald",
        "Admission Date": "2024/05/09",
        "State Code": "il",
        "Department Name": "Neurology",
        "Patient Age": 29,
        "Total Bill Amount ($)": "$3,100.00",
        "Insurance Covered?": "0",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        # Intentional exact duplicate
        "Patient Record #": "MED-8805",
        "Patient Full Name": "clara oswald",
        "Admission Date": "2024/05/09",
        "State Code": "il",
        "Department Name": "Neurology",
        "Patient Age": 29,
        "Total Bill Amount ($)": "$3,100.00",
        "Insurance Covered?": "0",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        # Fully empty row (all None)
        "Patient Record #": None,
        "Patient Full Name": None,
        "Admission Date": None,
        "State Code": None,
        "Department Name": None,
        "Patient Age": None,
        "Total Bill Amount ($)": None,
        "Insurance Covered?": None,
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8806",
        "Patient Full Name": "  victor FRIES ",
        "Admission Date": "11-06-2024",
        "State Code": "ny",
        "Department Name": None,  # Missing department
        "Patient Age": None,  # Missing age
        "Total Bill Amount ($)": "$7,800.00",
        "Insurance Covered?": "FALSE",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8807",
        "Patient Full Name": "HELEN CHO",
        "Admission Date": "2024-07-19",
        "State Code": None,  # Missing state code
        "Department Name": "  orthopedics  ",
        "Patient Age": 41,
        "Total Bill Amount ($)": "$2,350.00",
        "Insurance Covered?": "yes",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8808",
        "Patient Full Name": "alberto falcone",
        "Admission Date": "2024/08/30",
        "State Code": "ca",
        "Department Name": "cardiology",
        "Patient Age": -2,  # Invalid negative age
        "Total Bill Amount ($)": "$5,120.00",
        "Insurance Covered?": "No",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8809",
        "Patient Full Name": "barbara gordon",
        "Admission Date": "14-09-2024",
        "State Code": "tx",
        "Department Name": "physical therapy",
        "Patient Age": 26,
        "Total Bill Amount ($)": "$1,150.00",
        "Insurance Covered?": "y",
        "Doctor Discharge Notes (Empty)": None,
    },
    {
        "Patient Record #": "MED-8810",
        "Patient Full Name": "walter bishop",
        "Admission Date": "2024-10-24",
        "State Code": "fl",
        "Department Name": "  research & oncology ",
        "Patient Age": 66,
        "Total Bill Amount ($)": "$8,900.00",
        "Insurance Covered?": "n",
        "Doctor Discharge Notes (Empty)": None,
    },
]

df_excel = pd.DataFrame(excel_records)
excel_path = output_dir / "hospital_patient_admissions_dirty.xlsx"
df_excel.to_excel(excel_path, index=False, engine="openpyxl")
print(f"Created Excel: {excel_path.resolve()}")
