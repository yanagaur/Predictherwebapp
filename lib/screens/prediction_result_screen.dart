import 'package:flutter/material.dart';

class PredictionResultScreen extends StatelessWidget {
  final List<dynamic> predictions;
  final double bmi_a;
  final double bmi_b;
  final double bmi_c;

  PredictionResultScreen({
    required this.predictions,
    required this.bmi_a,
    required this.bmi_b,
    required this.bmi_c,
  });

  @override
  Widget build(BuildContext context) {
    bool isProneToPCOD = predictions.contains(1);
    String predictionMessage = isProneToPCOD
        ? 'You are prone to PCOD. Take care.'
        : 'You are safe, but take care of yourself.';

    List<String> adviceLines = isProneToPCOD
        ? [
      "Your body is signaling the need for extra care. Stay proactive to manage PCOD risks effectively.",
      "With the right approach, you can minimize the impact of PCOD. Prioritize your health and well-being.",
      "A healthy lifestyle can make all the difference. Take charge of your health to manage the risk of PCOD.",
    ]
        : [
      "Your current health is a testament to your good habits. Keep up the great work and stay vigilant.",
      "You’re in a good place, but maintaining it requires continuous effort. Keep taking care of yourself.",
      "Your well-being is your greatest asset. Continue nurturing it with mindful choices.",
    ];

    List<String> tips = isProneToPCOD
        ? [
      "Balanced Diet: Focus on a diet that includes whole grains, lean proteins, and plenty of fruits and vegetables. Avoid processed foods and reduce your intake of sugar and refined carbs.",
      "Regular Exercise: Engage in at least 30 minutes of moderate exercise most days of the week. Activities like walking, cycling, and swimming can help regulate weight and insulin levels.",
      "Stress Management: Practice mindfulness, meditation, or yoga to reduce stress, which can exacerbate PCOD symptoms.",
      "Sleep Well: Aim for 7-8 hours of quality sleep each night. Poor sleep can impact hormones that regulate appetite and metabolism.",
      "Regular Monitoring: Keep track of any symptoms and consider regular health check-ups to monitor your hormonal levels and overall health.",
      "Consult a Healthcare Provider: It’s important to consult your nearby doctor for personalized advice and to discuss possible treatment options or lifestyle changes that can help manage your condition effectively.",
    ]
        : [
      "Preventive Care: Regular check-ups with your doctor are essential, even when you feel well. They can catch potential issues early and help you stay on track with your health goals.",
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Prediction Results',
          style: TextStyle(color: Colors.pink.shade700, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        iconTheme: IconThemeData(color: Colors.pink.shade700),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Icon(
                  isProneToPCOD ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                  color: isProneToPCOD ? Colors.redAccent : Colors.green,
                  size: 80,
                ),
              ),
              SizedBox(height: 20),
              Center(
                child: Text(
                  predictionMessage,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: isProneToPCOD ? Colors.redAccent : Colors.green,
                  ),
                ),
              ),
              SizedBox(height: 30),
              Card(
                color: Colors.white,
                elevation: 6,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
                shadowColor: Colors.black.withOpacity(0.2),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: adviceLines.map((line) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 12.0),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.favorite, size: 20, color: Colors.pinkAccent),
                            SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                line,
                                style: TextStyle(fontSize: 16, color: Colors.black87),
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
              SizedBox(height: 30),
              Text(
                'Tips for You',
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: Colors.pink.shade700,
                  shadows: [
                    Shadow(
                      blurRadius: 2.0,
                      color: Colors.black26,
                      offset: Offset(1.0, 1.0),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 15),
              ...tips.map((tip) => Padding(
                padding: const EdgeInsets.only(bottom: 12.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.arrow_forward_ios, size: 22, color: Colors.pink.shade700),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        tip,
                        style: TextStyle(fontSize: 16, color: Colors.black87),
                      ),
                    ),
                  ],
                ),
              )),
            ],
          ),
        ),
      ),
    );
  }
}
