# data_fetcher.py
import pandas as pd
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from logger.logger import get_logger

logger = get_logger(__file__)


# Hard-coded database credentials
db_config = {
    'user': 'admin',
    'password': 'admin',
    'host': '192.168.1.37',
    'port': 3307,
    'database': 'bgi_jsrsbeta_final'
}

# Create connection string and engine
connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(connection_string)
logger.info(f"Database engine created for {db_config['host']}:{db_config['port']}/{db_config['database']}")

# SQL query to fetch product data
PRODUCT_QUERY = """
SELECT 
    pd.MemCompProdDtls_Pk as product_id, 
    pd.MCPrD_DisplayName AS product_name,
    pd.MCPrD_ProdDesc as description,
    c.bicc_categoryname,
    sc.bicsc_subcategoryname,
    p.bicpm_productname,
    cm.MCM_CompanyName,
    cy.CyM_CountryName,
    st.SM_StateName,
    ct.CM_CityName
FROM 
    memcompproddtls_tbl pd
LEFT JOIN bgiindcodecateg_tbl c ON pd.mcprd_bgiindcodecateg_fk = c.bgiindcodecateg_pk
LEFT JOIN bgiindcodesubcateg_tbl sc ON pd.mcprd_bgiindcodesubcateg_fk = sc.bgiindcodesubcateg_pk
LEFT JOIN bgiinduscodeprodmst_tbl p ON pd.mcprd_bgiinduscodeprodmst_fk = p.bgiinduscodeprodmst_pk
JOIN membercompanymst_tbl cm ON pd.MCPrD_MemberCompMst_Fk = cm.MemberCompMst_Pk
JOIN countrymst_tbl cy ON cm.MCM_CountryMst_Fk = cy.CountryMst_Pk
JOIN statemst_tbl st ON cm.MCM_StateMst_Fk = st.StateMst_Pk
JOIN citymst_tbl ct ON cm.MCM_CityMst_Fk = ct.CityMst_Pk;
"""

def test_connection():
    """Test the database connection and return True if successful."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

def fetch_data(query=PRODUCT_QUERY, chunksize=10000, max_retries=3):
    """Fetches data from the database using the given SQL query in chunks."""
    logger.info("Connecting to database and fetching data...")
    chunks = []  # List to store individual chunks
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            with engine.connect() as connection:
                # Use pd.read_sql with chunksize to fetch data in smaller chunks
                for i, chunk in enumerate(pd.read_sql(query, connection, chunksize=chunksize)):
                    chunks.append(chunk)  # Append each chunk to the list
                    logger.debug(f"Fetched chunk {i+1} with {len(chunk)} rows")
            
            # Concatenate all chunks into a single DataFrame
            data = pd.concat(chunks, ignore_index=True)
            logger.info(f"Data fetched successfully. Total rows: {len(data)}")
            
            # Basic data cleaning to avoid obvious issues
            # Replace None and null values with NaN for consistency
            data = data.replace([None, 'None', 'null', 'NULL'], pd.NA)
            
            # Strip whitespace from string columns
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].astype(str).str.strip()
                
            return data
        except SQLAlchemyError as e:
            retry_count += 1
            logger.warning(f"Database error (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count <= max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries exceeded. Failed to fetch data.")
                raise
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

if __name__ == "__main__":
    # Simple test script to check the connection and data fetching
    if test_connection():
        print("Database connection successful!")
        try:
            df = fetch_data()
            print(f"Successfully fetched {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            print("\nSample data:")
            print(df.head(3))
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
    else:
        print("Failed to connect to database. Check credentials and network.")