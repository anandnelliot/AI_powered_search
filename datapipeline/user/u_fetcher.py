# user_fetcher.py
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

# SQL query to fetch user data
USER_QUERY = """
SELECT
    u.UserMst_Pk as user_id,
    u.UM_LoginId as login,
    CONCAT(u.um_firstname, ' ', u.um_lastname) as user_name,
    u.UM_EmailID as email_id,
    u.UM_Designation_new as designation,
    mc.MCM_CompanyName as supplier,
    cy.CyM_CountryName as country,
    st.SM_StateName as state,
    ct.CM_CityName as city
FROM 
    usermst_tbl u
LEFT JOIN memberregistrationmst_tbl mr ON mr.MemberRegMst_Pk = u.UM_MemberRegMst_Fk
LEFT JOIN membercompanymst_tbl mc ON mr.MemberRegMst_Pk = mc.MCM_MemberRegMst_Fk
LEFT JOIN countrymst_tbl cy ON u.um_countrymst_fk = cy.CountryMst_Pk
LEFT JOIN statemst_tbl st ON u.um_statemst_fk = st.StateMst_Pk
LEFT JOIN citymst_tbl ct ON u.um_citymst_fk = ct.CityMst_Pk
"""

def fetch_data(query=USER_QUERY, chunksize=5000):
    """Fetches user data from the database using the given SQL query in chunks."""
    logger.info("Connecting to database and fetching user data...")
    chunks = []  # List to store individual chunks
    try:
        with engine.connect() as connection:
            # Use pd.read_sql with chunksize to fetch data in smaller chunks
            for i, chunk in enumerate(pd.read_sql(query, connection, chunksize=chunksize)):
                chunks.append(chunk)  # Append each chunk to the list
                logger.debug(f"Fetched chunk {i+1} with {len(chunk)} rows")
        
        # Concatenate all chunks into a single DataFrame
        data = pd.concat(chunks, ignore_index=True)
        logger.info(f"User data fetched successfully. Total rows: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error fetching user data: {str(e)}")
        raise