sqlite3 :memory: '.mode csv' \
    '.import "./archive/NFL Play by Play 2009-2018 (v5).csv" nfl' \
    '.read play_data.sql' > play_data.csv
