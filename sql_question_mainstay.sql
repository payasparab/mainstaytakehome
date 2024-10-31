-- Number 1
/* 
My assumption here is that "were listed" implies that they were at some point listed.
IF you wanted to look at the homes that are currently listed and when they were listed you would add 
AND home_state = 'listed' to the WHERE clause
*/

SELECT 
    DATE_FORMAT(list_date, '%Y-%m') AS listing_month,
    market_name,
    COUNT(home_id) AS total_listed_homes
FROM 
    analytics.homes
WHERE 
    list_date IS NOT NULL
GROUP BY 
    listing_month, market_name
ORDER BY 
    listing_month, market_name;

-- Number 2 
WITH listed_homes AS (
    SELECT 
        home_id,
        DATE_FORMAT(list_date, '%Y-%m') AS listing_month,
        market_name
    FROM 
        analytics.homes
    WHERE 
        list_date IS NOT NULL
), 
homes_with_offers AS (
    SELECT DISTINCT 
        o.home_id,
        lh.listing_month,
        lh.market_name
    FROM 
        analytics.listing_offers AS o
    JOIN 
        listed_homes AS lh ON o.home_id = lh.home_id
),
offers_flagged AS (
    SELECT 
        lh.home_id,
        lh.listing_month,
        lh.market_name,
        CASE WHEN hwo.home_id IS NOT NULL THEN 1 ELSE 0 END AS received_offer
    FROM 
        listed_homes AS lh
    LEFT JOIN 
        homes_with_offers AS hwo ON lh.home_id = hwo.home_id
)
SELECT 
    listing_month,
    market_name,
    SUM(received_offer) * 100.0 / COUNT(*) AS offer_percentage
FROM 
    offers_flagged
GROUP BY 
    listing_month, market_name
ORDER BY 
    listing_month, market_name;

-- Number 3
WITH listed_homes AS (
    SELECT 
        home_id,
        DATE_FORMAT(list_date, '%Y-%m') AS listing_month,
        market_name,
  		list_date
    FROM 
        analytics.homes
    WHERE 
        list_date IS NOT NULL
), first_7_days_visits AS (
    SELECT 
        lh.home_id,
        lh.listing_month,
        lh.market_name,
        COUNT(DISTINCT hv.customer_id) AS unique_visitors_7_days
    FROM 
        listed_homes AS lh
    JOIN 
        analytics.home_visits AS hv ON lh.home_id = hv.home_id
    WHERE 
        hv.visited_at BETWEEN lh.list_date AND DATE_ADD(lh.list_date, INTERVAL 7 DAY)
    GROUP BY 
        lh.home_id, lh.listing_month, lh.market_name
)
SELECT 
    listing_month,
    market_name,
    AVG(unique_visitors_7_days) AS avg_unique_visitors_first_7_days
FROM 
    first_7_days_visits
GROUP BY 
    listing_month, market_name
ORDER BY 
    listing_month, market_name;