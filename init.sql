CREATE TABLE IF NOT EXISTS campaigns
(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) CHECK (name != ''),
    channel VARCHAR(100) CHECK (channel != ''),
    spend NUMERIC(10, 2),
    impressions INT,
    clicks INT,
    conversions INT,
    month VARCHAR(100) CHECK (month != '')
);

CREATE TABLE IF NOT EXISTS traffic
(
    id SERIAL PRIMARY KEY,
    source VARCHAR(100) CHECK (source != ''),
    sessions INT,
    bounce_rate NUMERIC(10, 2),
    avg_duration_seconds INT,
    month VARCHAR(100) CHECK (month != '')
);

CREATE TABLE IF NOT EXISTS content
(
    id SERIAL PRIMARY KEY,
    title VARCHAR(100) CHECK (title != ''),
    type VARCHAR(100) CHECK (type != ''),
    views INT,
    shares INT,
    leads_generated INT,
    published_date DATE
);

-- Sample campaigns
INSERT INTO campaigns (name, channel, spend, impressions, clicks, conversions, month) VALUES
('Summer Sale',        'Google Ads',  5000, 180000, 12000, 340, '2024-06'),
('Brand Awareness',    'Instagram',   3000, 250000,  8500, 120, '2024-06'),
('Email Re-engage',    'Email',        500,  42000,  4200, 210, '2024-06'),
('Retargeting Push',   'Facebook',    2000,  95000,  6100, 180, '2024-07'),
('SEO Content Push',   'Organic',        0, 320000, 22000, 430, '2024-07');

-- Sample traffic
INSERT INTO traffic (source, sessions, bounce_rate, avg_duration_seconds, month) VALUES
('Organic Search', 32000, 42.5, 185, '2024-06'),
('Paid Search',    12000, 58.2, 102, '2024-06'),
('Social Media',    8500, 65.1,  78, '2024-06'),
('Direct',          5200, 35.0, 210, '2024-06'),
('Email',           4100, 28.5, 240, '2024-06');

-- Sample content
INSERT INTO content (title, type, views, shares, leads_generated, published_date) VALUES
('10 Ways to Grow Your Email List',  'Blog Post',  15200, 340, 85,  '2024-05-10'),
('Product Demo Walkthrough',         'Video',      28000, 120, 42,  '2024-05-18'),
('Q2 Industry Report',               'Whitepaper',  3400,  90, 210, '2024-06-01'),
('How We Grew 3x in 6 Months',       'Case Study',  9800, 280, 130, '2024-06-15');