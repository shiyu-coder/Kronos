"use client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

import InfoPanel from "./InfoPanel";
import PredictionChart from "./PredictionChart";

const Dashboard = () => {
  return (
    <section className='size-full border'>
      <Card className='size-full'>
        <CardHeader>
          <InfoPanel />
        </CardHeader>

        <CardContent className=''>
          <div className='flex flex-col gap-4'>
            <div className='flex flex-col gap-0'>
              <h2 className='text-base font-semibold'>Prediction preview</h2>
              <p className='text-sm/relaxed text-muted-foreground'>
                History candles transition into the forecast region.
              </p>
            </div>

            <PredictionChart />
          </div>
        </CardContent>
      </Card>
    </section>
  );
};

export default Dashboard;
