"""
constants.py
------------
Shared constants for the Intrusion Detection System project.

The NSL-KDD dataset ships as plain CSV/text files with NO header row, so the
41 feature names + 1 label column have to be defined here and reused by every
script that touches the data (preprocessing, training, and the Streamlit app)
so that column order always matches.
"""

# The 41 standard NSL-KDD features, in the exact order they appear in the
# raw KDDTrain+ / KDDTest+ files, followed by the "label" column.
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]

# Some distributions of NSL-KDD include a 43rd "difficulty level" column at
# the end. We detect and drop it automatically in preprocessing if present.

# The three categorical (text) columns that need label/one-hot encoding.
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# A reasonably complete set of values seen across NSL-KDD train + test for
# each categorical column. Used to build encoders that won't break if the
# test set contains a service/flag value absent from the training set.
PROTOCOL_TYPES = ["tcp", "udp", "icmp"]

FLAG_VALUES = ["SF", "S0", "REJ", "RSTR", "SH", "RSTO", "S1", "RSTOS0", "S3", "S2", "OTH"]

SERVICE_VALUES = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard",
    "domain", "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger",
    "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784",
    "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin", "kshell",
    "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns",
    "netbios_ssn", "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump",
    "pop_2", "pop_3", "printer", "private", "red_i", "remote_job", "rje",
    "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat", "telnet",
    "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet",
    "whois", "X11", "Z39_50",
]

# Mapping from the ~22 raw NSL-KDD attack labels into the 4 well-known
# attack CATEGORIES, used only for the richer multi-class info shown in the
# app / EDA notebook. The actual ML models in this project are trained as a
# BINARY classifier: normal (0) vs attack (1), which is what the project
# brief calls for ("safe" vs "suspicious").
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "apache2": "dos",
    "processtable": "dos", "udpstorm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",
    # R2L (remote to local)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l",
    "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l",
    "snmpgetattack": "r2l", "httptunnel": "r2l", "sendmail": "r2l",
    "named": "r2l",
    # U2R (user to root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}

RANDOM_STATE = 42
