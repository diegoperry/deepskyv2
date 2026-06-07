create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  free_credits_remaining integer not null default 3 check (free_credits_remaining >= 0),
  mars_code_redeemed boolean not null default false,
  subscription_status text not null default 'free',
  stripe_customer_id text unique,
  stripe_subscription_id text,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

alter table public.profiles
add column if not exists mars_code_redeemed boolean not null default false;

do $$
begin
  if not exists (
    select 1
    from pg_policies
    where schemaname = 'public'
      and tablename = 'profiles'
      and policyname = 'Users can read their own profile'
  ) then
    create policy "Users can read their own profile"
    on public.profiles
    for select
    to authenticated
    using (auth.uid() = user_id);
  end if;
end;
$$;

create or replace function public.consume_free_credit(target_user_id uuid)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  did_consume boolean;
begin
  update public.profiles
  set
    free_credits_remaining = free_credits_remaining - 1,
    updated_at = now()
  where
    user_id = target_user_id
    and free_credits_remaining > 0
    and coalesce(subscription_status, 'free') not in ('active', 'trialing')
  returning true into did_consume;

  return coalesce(did_consume, false);
end;
$$;

create or replace function public.redeem_mars_code(target_user_id uuid)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  did_redeem boolean;
begin
  update public.profiles
  set
    free_credits_remaining = free_credits_remaining + 5,
    mars_code_redeemed = true,
    updated_at = now()
  where
    user_id = target_user_id
    and mars_code_redeemed = false
    and coalesce(subscription_status, 'free') not in ('active', 'trialing')
  returning true into did_redeem;

  return coalesce(did_redeem, false);
end;
$$;
