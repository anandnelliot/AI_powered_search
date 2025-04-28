# su_fetcher.py
import logging
import pandas as pd
from sqlalchemy import create_engine, text

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

# SQL query to fetch supplier data with products and services as comma-separated lists
SUPPLIER_QUERY = """
SELECT 
    cm.MemberCompMst_Pk as supplier_id,
    cm.MCM_CompanyName as supplier_name,
    c.ClM_ClassificationType as classification,
    GROUP_CONCAT(DISTINCT s.SecM_SectorName SEPARATOR ', ') as sector,
    mr.mrm_businsrc as businesssource,
    cy.CyM_CountryName as country,
    st.SM_StateName as state,
    ct.CM_CityName as city,
    GROUP_CONCAT(DISTINCT COALESCE(pd.MCPrD_DisplayName, sc.MCSvD_DisplayName) SEPARATOR ', ') as products_and_services
FROM 
    membercompanymst_tbl cm
LEFT JOIN memcompproddtls_tbl pd ON pd.MCPrD_MemberCompMst_Fk = cm.MemberCompMst_Pk 
LEFT JOIN memcompservicedtls_tbl sc ON sc.MCSvD_MemberCompMst_Fk = cm.MemberCompMst_Pk
LEFT JOIN countrymst_tbl cy ON cm.MCM_CountryMst_Fk = cy.CountryMst_Pk
LEFT JOIN statemst_tbl st ON cm.MCM_StateMst_Fk = st.StateMst_Pk
LEFT JOIN citymst_tbl ct ON cm.MCM_CityMst_Fk = ct.CityMst_Pk
LEFT JOIN memberregistrationmst_tbl mr ON mr.MemberRegMst_Pk = cm.MCM_MemberRegMst_Fk
LEFT JOIN classificationmst_tbl c ON c.ClassificationMst_Pk = cm.mcm_classificationmst_fk
LEFT JOIN sectormst_tbl s ON FIND_IN_SET(s.SectorMst_Pk, mr.mrm_compsector)
GROUP BY 
    cm.MemberCompMst_Pk, 
    cm.MCM_CompanyName, 
    c.ClM_ClassificationType,
    mr.mrm_businsrc,
    cy.CyM_CountryName, 
    st.SM_StateName, 
    ct.CM_CityName;
"""

def fetch_data(query=SUPPLIER_QUERY, chunksize=5000):
    """Fetches supplier data from the database using the given SQL query in chunks."""
    logger.info("Connecting to database and fetching supplier data...")
    chunks = []  # List to store individual chunks
    try:
        with engine.connect() as connection:
            # Use pd.read_sql with chunksize to fetch data in smaller chunks
            for i, chunk in enumerate(pd.read_sql(query, connection, chunksize=chunksize)):
                chunks.append(chunk)  # Append each chunk to the list
                logger.debug(f"Fetched chunk {i+1} with {len(chunk)} rows")
        
        # Concatenate all chunks into a single DataFrame
        data = pd.concat(chunks, ignore_index=True)
        logger.info(f"Supplier data fetched successfully. Total rows: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error fetching supplier data: {str(e)}")
        raise

