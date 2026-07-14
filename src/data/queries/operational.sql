create table if not exists {target_table} as (
    
    with table0 as (
        select b.*, a.{name}
        from {weather_table} as b join {temp_table} as a
        on a.{id} = b.{id} and a.{date} = b.{date} and a.{hour} = b.{hour}
    ),

    table1 as (
        select a.*, b.{value}
        from table0 as a join {bar_table} as b
        on a.{date} = b.{date} and a.{hour} = b.{hour}
    ),

    table2 as (
        select t.*, l.{forecast}
        from table1 as t join {load_table} as l
        on t.{date} = l.{date} and t.{hour} = l.{hour}
    ),

    table3 as (
        select a.*, b.{generation}, b.{code}
        from table2 as a join {energy_table} as b
        on a.{id} = b.{id} and a.{date} = b.{date} and a.{hour} = b.{hour} 
    ),

    table4 as (
        select t.*, d.{declare}
        from table3 as t join {seller_table} as d
        on t.{id} = d.{id} and t.{code} = d.{code} and t.{date} = d.{date} and t.{hour} = d.{hour}
    ),

    table5 as (
        select z.*, v.{status_type}
        from table4 as z join {status_table} as v
        on z.{id} = v.{id} and z.{code} = v.{code} and z.{date} = v.{date} and z.{hour} = v.{hour}
    ),
    
    table6 as (
        select m.*, c.{require}
        from table5 as m join {commitment_table} as c
        on m.{id} = c.{id} and m.{code} = c.{code} and m.{date} = c.{date} and m.{hour} = c.{hour}
    )

    select * from table6
)


CREATE TABLE if not exists integrated_data AS
SELECT
    c.id,
    c.name,
    c.code,
    c.date,
    c.hour,

    w.temperature,
    w.humidity,
    w.dew,
    w.apparent_temperature,
    w.precipitation,
    w.rain,
    w.snow,
    w.surface_pressure,
    w.evapotransporation,
    w.wind_speed,
    w.wind_direction,

    b.value,
    l.forecast,
    e.generation,
    s.declare,
    st.status,
    c.require

FROM commitment c

LEFT JOIN weather w 
    ON c.id = w.id AND c.date = w.date AND c.hour = w.hour

LEFT JOIN bar b 
    ON c.date = b.date AND c.hour = b.hour

LEFT JOIN load l 
    ON c.date = l.date AND c.hour = l.hour

LEFT JOIN energy e 
    ON c.id = e.id AND c.code = e.code AND c.date = e.date AND c.hour = e.hour

LEFT JOIN selleroffer s 
    ON c.id = s.id AND c.code = s.code AND c.date = s.date AND c.hour = s.hour

LEFT JOIN status st 
    ON c.id = st.id AND c.code = st.code AND c.date = st.date AND c.hour = st.hour;
