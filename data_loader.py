import pandas as pd
import io
import sqlite3
import tempfile
import os


def load_data(file) -> tuple:
    file_type = file.name.rsplit(".", 1)[-1].lower()

    try:
        file.seek(0)

        if file_type == "csv":
            df = pd.read_csv(file, low_memory=False)

        elif file_type == "tsv":
            df = pd.read_csv(file, sep="\t", low_memory=False)

        elif file_type in ("xlsx", "xls"):
            df = pd.read_excel(file)

        elif file_type == "json":
            raw = file.read()
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                df = pd.read_json(io.BytesIO(raw), lines=True)

        elif file_type == "txt":
            content = file.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(content), sep=None, engine="python")

        elif file_type == "html":
            tables = pd.read_html(file)
            if not tables:
                return None, "No tables found in HTML file."
            df = tables[0]

        elif file_type == "sql":
            content = file.read().decode("utf-8", errors="ignore")
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    tmp_path = tmp.name
                conn = sqlite3.connect(tmp_path)
                conn.executescript(content)
                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table';", conn
                )
                if tables.empty:
                    conn.close()
                    return None, "No tables found in SQL file."
                table_name = tables.iloc[0, 0]
                df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
                conn.close()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif file_type in ("db", "sqlite3"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table';", conn
                )
                if tables.empty:
                    conn.close()
                    return None, "No tables found in database file."
                table_name = tables.iloc[0, 0]
                df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
                conn.close()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        else:
            return None, f"Unsupported file format: '.{file_type}'"

        if df.empty:
            return None, "The file loaded successfully but contains no data."

        df.columns = [str(c) for c in df.columns]
        return df, None

    except Exception as exc:
        return None, str(exc)