import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';
import { createSettlementHandler } from './handler.mjs';

Deno.serve(createSettlementHandler({ env: (key: string) => Deno.env.get(key), createClient }));
