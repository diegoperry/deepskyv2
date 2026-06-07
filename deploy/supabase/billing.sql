create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  free_credits_remaining integer not null default 3 check (free_credits_remaining >= 0),
  subscription_status text not null default 'free',
  stripe_customer_id text unique,
  stripe_subscription_id text,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

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

create or replace function public.refund_free_credit(target_user_id uuid)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  did_refund boolean;
begin
  update public.profiles
  set
    free_credits_remaining = least(free_credits_remaining + 1, 3),
    updated_at = now()
  where
    user_id = target_user_id
    and free_credits_remaining < 3
    and coalesce(subscription_status, 'free') not in ('active', 'trialing')
  returning true into did_refund;

  return coalesce(did_refund, false);
end;
$$;
