# data_fetcher.py
import pandas as pd
from sqlalchemy import create_engine
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

# SQL query to fetch service data
SERVICE_QUERY = """
SELECT 
    sd.MemCompServDtls_Pk as service_id, 
    sd.MCSvD_DisplayName AS service_name,
    sd.MCSvD_ServDesc,
    c.bicc_categoryname,
    sc.bicsc_subcategoryname,
    s.bicsm_servicename,
    cm.MCM_CompanyName,
    cy.CyM_CountryName,
    st.SM_StateName,
    ct.CM_CityName
FROM 
    memcompservicedtls_tbl sd
LEFT JOIN bgiindcodecateg_tbl c ON sd.mcsvd_bgiindcodecateg_fk = c.bgiindcodecateg_pk
LEFT JOIN bgiindcodesubcateg_tbl sc ON sd.mcsvd_bgiindcodesubcateg_fk = sc.bgiindcodesubcateg_pk
LEFT JOIN bgiinduscodeservmst_tbl s ON sd.mcsvd_bgiinduscodeservmst_fk = s.bgiinduscodeservmst_pk
JOIN membercompanymst_tbl cm ON sd.MCSvD_MemberCompMst_Fk = cm.MemberCompMst_Pk
JOIN countrymst_tbl cy ON cm.MCM_CountryMst_Fk = cy.CountryMst_Pk
JOIN statemst_tbl st ON cm.MCM_StateMst_Fk = st.StateMst_Pk
JOIN citymst_tbl ct ON cm.MCM_CityMst_Fk = ct.CityMst_Pk;
"""

def fetch_data(query=SERVICE_QUERY, chunksize=10000):
    """Fetches data from the database using the given SQL query in chunks."""
    logger.info("Connecting to database and fetching service data...")
    chunks = []  # List to store individual chunks
    try:
        with engine.connect() as connection:
            # Use pd.read_sql with chunksize to fetch data in smaller chunks
            for i, chunk in enumerate(pd.read_sql(query, connection, chunksize=chunksize)):
                chunks.append(chunk)  # Append each chunk to the list
                logger.debug(f"Fetched chunk {i+1} with {len(chunk)} rows")
        
        # Concatenate all chunks into a single DataFrame
        data = pd.concat(chunks, ignore_index=True)
        logger.info(f"Service data fetched successfully. Total rows: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise