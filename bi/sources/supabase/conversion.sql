-- Free → paid: distinct paying users vs total registered users (single row).
select
  (select count(*) from profiles) as total_users,
  (select count(distinct user_id)
     from credit_transactions
    where reason = 'stripe_purchase') as paying_users
